# 02-BANKING-ENGINE-PLATFORM/payments-network/instant-settlement/settlement_engine_pro.py
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from enum import Enum
import hashlib
import struct
import base64
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import numpy as np
import pandas as pd
from fastapi import FastAPI, APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Numeric, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
import aiomysql
import motor.motor_asyncio
from prometheus_client import Counter, Histogram, Gauge, Summary
import logging
from logging.handlers import RotatingFileHandler
import jwt
from typing_extensions import Annotated

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('settlement_engine.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
SETTLEMENT_REQUESTS = Counter('settlement_requests_total', 'Total settlement requests')
SETTLEMENT_SUCCESS = Counter('settlement_success_total', 'Successful settlements')
SETTLEMENT_FAILURE = Counter('settlement_failure_total', 'Failed settlements', ['reason'])
SETTLEMENT_LATENCY = Histogram('settlement_latency_seconds', 'Settlement latency')
SETTLEMENT_AMOUNT = Gauge('settlement_amount_usd', 'Settlement amount in USD')
ACTIVE_SETTLEMENTS = Gauge('active_settlements', 'Currently active settlements')
QUEUE_SIZE = Gauge('settlement_queue_size', 'Settlement queue size')

# Database models
Base = declarative_base()

class SettlementTransaction(Base):
    __tablename__ = "settlement_transactions"
    
    id = Column(String(64), primary_key=True)
    reference_id = Column(String(128), unique=True, index=True)
    status = Column(String(32), index=True)
    from_account = Column(String(128), index=True)
    to_account = Column(String(128), index=True)
    amount = Column(Numeric(24, 8))
    currency = Column(String(3))
    settlement_currency = Column(String(3))
    fx_rate = Column(Numeric(24, 12))
    fees = Column(Numeric(24, 8))
    net_amount = Column(Numeric(24, 8))
    created_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime)
    settlement_time_ms = Column(Numeric(12, 3))
    quantum_signature = Column(Text)
    quantum_proof = Column(Text)
    compliance_status = Column(String(32))
    risk_score = Column(Numeric(5, 2))
    error_message = Column(Text)
    metadata = Column(JSON)
    trace_id = Column(String(64))
    parent_transaction_id = Column(String(64))
    retry_count = Column(Numeric(3, 0), default=0)
    last_retry_at = Column(DateTime)

# Pydantic models
class PaymentRequest(BaseModel):
    from_account: str = Field(..., description="Source account identifier")
    to_account: str = Field(..., description="Destination account identifier")
    amount: Decimal = Field(..., gt=0, description="Payment amount")
    currency: str = Field(..., regex="^[A-Z]{3}$", description="Currency code (ISO 4217)")
    settlement_currency: str = Field(None, regex="^[A-Z]{3}$", description="Settlement currency")
    reference: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    priority: str = Field("instant", regex="^(instant|priority|standard)$")
    quantum_secure: bool = Field(True)
    compliance_data: Optional[Dict] = None
    metadata: Optional[Dict] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class PaymentResponse(BaseModel):
    transaction_id: str
    status: str
    settlement_time_ms: float
    fx_rate: Optional[Decimal] = None
    fees: Decimal
    net_amount: Decimal
    quantum_proof: Optional[Dict] = None
    compliance_status: str
    estimated_arrival: datetime
    trace_id: str

class BatchPaymentRequest(BaseModel):
    payments: List[PaymentRequest]
    batch_reference: str = Field(default_factory=lambda: str(uuid.uuid4()))
    atomic_settlement: bool = Field(True)
    notification_url: Optional[str] = None

class BatchPaymentResponse(BaseModel):
    batch_id: str
    total_payments: int
    successful: int
    failed: int
    total_amount: Decimal
    total_fees: Decimal
    settlement_time_ms: float
    results: List[Dict]

# Settlement status enum
class SettlementStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SETTLED = "settled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLIANCE_HOLD = "compliance_hold"
    RISK_HOLD = "risk_hold"

# Quantum settlement engine
class QuantumSettlementEngine:
    """Complete quantum-secure instant settlement engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis = None
        self.db = None
        self.http_session = None
        self.fx_provider = FXProvider(config['fx_provider'])
        self.compliance_engine = ComplianceEngine(config['compliance'])
        self.risk_engine = RiskEngine(config['risk'])
        self.quantum_crypto = QuantumCryptoService(config['quantum_crypto'])
        self.liquidity_pool = LiquidityPool(config['liquidity'])
        self.metrics_collector = MetricsCollector()
        
        # Queue management
        self.pending_settlements = asyncio.Queue(maxsize=10000)
        self.processing_semaphore = asyncio.Semaphore(100)  # Max concurrent settlements
        
        # Performance optimization
        self.batch_processor = BatchProcessor()
        self.cache = {}
        self.circuit_breaker = CircuitBreaker()
        
        # Start background tasks
        self.background_tasks = []
        
    async def initialize(self):
        """Initialize engine components"""
        # Initialize Redis
        self.redis = await redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            password=self.config['redis'].get('password'),
            decode_responses=True
        )
        
        # Initialize database
        engine = create_async_engine(self.config['database']['url'])
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        self.db = async_session
        
        # Initialize HTTP session
        self.http_session = aiohttp.ClientSession()
        
        # Start background processors
        self.background_tasks.extend([
            asyncio.create_task(self._process_settlement_queue()),
            asyncio.create_task(self._monitor_performance()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._reconcile_settlements())
        ])
        
        logger.info("Quantum Settlement Engine initialized")
    
    async def process_payment(self, payment_request: PaymentRequest) -> PaymentResponse:
        """Process a single payment with quantum security"""
        SETTLEMENT_REQUESTS.inc()
        ACTIVE_SETTLEMENTS.inc()
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        try:
            # Step 1: Validate and enrich payment
            enriched_payment = await self._validate_and_enrich_payment(payment_request, trace_id)
            
            # Step 2: Compliance check
            compliance_result = await self.compliance_engine.check_payment(enriched_payment)
            if not compliance_result['approved']:
                await self._handle_compliance_hold(enriched_payment, compliance_result)
                raise HTTPException(403, f"Compliance check failed: {compliance_result['reason']}")
            
            # Step 3: Risk assessment
            risk_assessment = await self.risk_engine.assess_payment(enriched_payment)
            enriched_payment['risk_score'] = risk_assessment['score']
            
            if risk_assessment['block']:
                await self._handle_risk_hold(enriched_payment, risk_assessment)
                raise HTTPException(400, f"Risk assessment failed: {risk_assessment['reason']}")
            
            # Step 4: FX rate calculation
            fx_data = await self._get_fx_rate(
                enriched_payment['currency'],
                enriched_payment.get('settlement_currency', enriched_payment['currency'])
            )
            enriched_payment['fx_rate'] = fx_data['rate']
            enriched_payment['fx_timestamp'] = fx_data['timestamp']
            
            # Step 5: Fee calculation
            fees = await self._calculate_fees(enriched_payment)
            enriched_payment['fees'] = fees
            enriched_payment['net_amount'] = enriched_payment['amount'] - fees
            
            # Step 6: Quantum signature
            if payment_request.quantum_secure:
                quantum_signature = await self.quantum_crypto.sign_transaction(enriched_payment)
                enriched_payment['quantum_signature'] = quantum_signature
            
            # Step 7: Liquidity check and lock
            liquidity_lock = await self.liquidity_pool.lock_liquidity(
                enriched_payment['from_account'],
                enriched_payment['net_amount'],
                enriched_payment['currency']
            )
            
            if not liquidity_lock['success']:
                raise HTTPException(400, f"Insufficient liquidity: {liquidity_lock['reason']}")
            
            # Step 8: Atomic settlement execution
            settlement_result = await self._execute_atomic_settlement(enriched_payment)
            
            if settlement_result['success']:
                # Step 9: Generate quantum proof
                quantum_proof = await self.quantum_crypto.generate_proof(settlement_result)
                
                # Step 10: Update ledger
                await self._update_ledger(settlement_result)
                
                # Step 11: Compliance reporting
                await self._report_to_compliance(settlement_result)
                
                # Step 12: Release liquidity lock
                await self.liquidity_pool.release_lock(liquidity_lock['lock_id'])
                
                elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update metrics
                SETTLEMENT_SUCCESS.inc()
                SETTLEMENT_LATENCY.observe(elapsed_time / 1000)
                SETTLEMENT_AMOUNT.set(float(enriched_payment['amount']))
                
                response = PaymentResponse(
                    transaction_id=settlement_result['transaction_id'],
                    status="settled",
                    settlement_time_ms=elapsed_time,
                    fx_rate=Decimal(str(fx_data['rate'])),
                    fees=Decimal(str(fees)),
                    net_amount=Decimal(str(enriched_payment['net_amount'])),
                    quantum_proof=quantum_proof,
                    compliance_status="approved",
                    estimated_arrival=datetime.utcnow(),
                    trace_id=trace_id
                )
                
                logger.info(f"Payment settled: {response.transaction_id} in {elapsed_time:.2f}ms")
                
                # Send real-time notification
                asyncio.create_task(self._send_notification(enriched_payment, response))
                
                return response
            
            else:
                SETTLEMENT_FAILURE.labels(reason='settlement_failed').inc()
                raise HTTPException(500, f"Settlement failed: {settlement_result['error']}")
                
        except HTTPException:
            raise
        except Exception as e:
            SETTLEMENT_FAILURE.labels(reason='unexpected_error').inc()
            logger.error(f"Payment processing error: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Payment processing failed: {str(e)}")
        finally:
            ACTIVE_SETTLEMENTS.dec()
    
    async def process_batch_payments(self, batch_request: BatchPaymentRequest) -> BatchPaymentResponse:
        """Process batch payments with atomic settlement"""
        batch_start_time = time.time()
        batch_id = batch_request.batch_reference
        
        try:
            # Validate batch
            if len(batch_request.payments) > 1000:
                raise HTTPException(400, "Batch size exceeds maximum of 1000 payments")
            
            # Group payments for optimization
            grouped_payments = self._group_payments_by_currency(batch_request.payments)
            
            results = []
            successful = 0
            failed = 0
            total_amount = Decimal('0')
            total_fees = Decimal('0')
            
            # Process groups in parallel
            processing_tasks = []
            for currency_pair, payments in grouped_payments.items():
                task = asyncio.create_task(
                    self._process_payment_group(currency_pair, payments, batch_id)
                )
                processing_tasks.append(task)
            
            # Wait for all groups to complete
            group_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Aggregate results
            for result in group_results:
                if isinstance(result, Exception):
                    failed += len(result.args[0]) if hasattr(result, 'args') else 1
                    results.append({'status': 'failed', 'error': str(result)})
                else:
                    successful += result['successful']
                    failed += result['failed']
                    total_amount += Decimal(str(result['total_amount']))
                    total_fees += Decimal(str(result['total_fees']))
                    results.extend(result['results'])
            
            elapsed_time = (time.time() - batch_start_time) * 1000
            
            response = BatchPaymentResponse(
                batch_id=batch_id,
                total_payments=len(batch_request.payments),
                successful=successful,
                failed=failed,
                total_amount=total_amount,
                total_fees=total_fees,
                settlement_time_ms=elapsed_time,
                results=results
            )
            
            logger.info(f"Batch {batch_id} processed: {successful} successful, {failed} failed")
            
            # Send batch notification
            if batch_request.notification_url:
                asyncio.create_task(self._send_batch_notification(batch_request, response))
            
            return response
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Batch processing failed: {str(e)}")
    
    async def _process_settlement_queue(self):
        """Background task to process settlement queue"""
        while True:
            try:
                # Get next settlement from queue
                settlement_data = await self.pending_settlements.get()
                
                # Process with semaphore for concurrency control
                async with self.processing_semaphore:
                    await self._process_queued_settlement(settlement_data)
                
                self.pending_settlements.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_queued_settlement(self, settlement_data: Dict):
        """Process a settlement from the queue"""
        try:
            # Revalidate before processing
            revalidation = await self._revalidate_settlement(settlement_data)
            
            if not revalidation['valid']:
                await self._handle_failed_settlement(settlement_data, revalidation['reason'])
                return
            
            # Check circuit breaker
            if self.circuit_breaker.is_open():
                await self._delay_settlement(settlement_data)
                return
            
            # Execute settlement
            result = await self._execute_queued_settlement(settlement_data)
            
            if result['success']:
                # Update metrics
                self.metrics_collector.record_settlement_success(
                    result['transaction_id'],
                    result['processing_time']
                )
            else:
                # Handle retry logic
                await self._handle_settlement_retry(settlement_data, result['error'])
                
        except Exception as e:
            logger.error(f"Queued settlement processing error: {e}", exc_info=True)
            await self._handle_settlement_error(settlement_data, str(e))
    
    async def _execute_atomic_settlement(self, payment_data: Dict) -> Dict:
        """Execute atomic settlement with rollback protection"""
        transaction_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Begin atomic transaction
            async with self.db() as session:
                # Step 1: Debit source account
                debit_result = await self._debit_account(
                    session,
                    payment_data['from_account'],
                    payment_data['net_amount'],
                    payment_data['currency']
                )
                
                if not debit_result['success']:
                    raise Exception(f"Debit failed: {debit_result['error']}")
                
                # Step 2: Credit destination account
                credit_result = await self._credit_account(
                    session,
                    payment_data['to_account'],
                    payment_data['net_amount'],
                    payment_data['currency']
                )
                
                if not credit_result['success']:
                    raise Exception(f"Credit failed: {credit_result['error']}")
                
                # Step 3: Record transaction
                transaction_record = SettlementTransaction(
                    id=transaction_id,
                    reference_id=payment_data.get('reference', str(uuid.uuid4())),
                    status=SettlementStatus.SETTLED.value,
                    from_account=payment_data['from_account'],
                    to_account=payment_data['to_account'],
                    amount=payment_data['amount'],
                    currency=payment_data['currency'],
                    settlement_currency=payment_data.get('settlement_currency', payment_data['currency']),
                    fx_rate=payment_data.get('fx_rate', 1.0),
                    fees=payment_data['fees'],
                    net_amount=payment_data['net_amount'],
                    settled_at=datetime.utcnow(),
                    settlement_time_ms=(time.time() - start_time) * 1000,
                    quantum_signature=payment_data.get('quantum_signature'),
                    quantum_proof=payment_data.get('quantum_proof'),
                    compliance_status='approved',
                    risk_score=payment_data.get('risk_score', 0.0),
                    metadata=payment_data.get('metadata'),
                    trace_id=payment_data.get('trace_id')
                )
                
                session.add(transaction_record)
                await session.commit()
                
                processing_time = time.time() - start_time
                
                return {
                    'success': True,
                    'transaction_id': transaction_id,
                    'processing_time': processing_time,
                    'settlement_time': transaction_record.settled_at,
                    'amount': float(payment_data['net_amount']),
                    'currency': payment_data['currency']
                }
                
        except Exception as e:
            # Automatic rollback
            await self._rollback_settlement(payment_data)
            raise e
    
    async def _debit_account(self, session, account_id: str, amount: Decimal, currency: str) -> Dict:
        """Debit account with balance check"""
        # In production, this would interact with actual account system
        # For simulation, we'll use Redis
        try:
            account_key = f"account:{account_id}:{currency}"
            current_balance = await self.redis.get(account_key)
            
            if current_balance is None:
                current_balance = '1000000'  # Default balance for simulation
            
            current_balance = Decimal(current_balance)
            
            if current_balance < amount:
                return {
                    'success': False,
                    'error': 'Insufficient funds',
                    'current_balance': float(current_balance),
                    'requested_amount': float(amount)
                }
            
            new_balance = current_balance - amount
            await self.redis.set(account_key, str(new_balance))
            
            # Record debit transaction
            debit_record = {
                'account_id': account_id,
                'amount': float(amount),
                'currency': currency,
                'type': 'debit',
                'timestamp': datetime.utcnow().isoformat(),
                'new_balance': float(new_balance)
            }
            
            await self.redis.lpush(f"transactions:{account_id}", json.dumps(debit_record))
            
            return {
                'success': True,
                'new_balance': float(new_balance),
                'transaction_id': str(uuid.uuid4())
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _credit_account(self, session, account_id: str, amount: Decimal, currency: str) -> Dict:
        """Credit account"""
        try:
            account_key = f"account:{account_id}:{currency}"
            current_balance = await self.redis.get(account_key)
            
            if current_balance is None:
                current_balance = '0'
            
            current_balance = Decimal(current_balance)
            new_balance = current_balance + amount
            
            await self.redis.set(account_key, str(new_balance))
            
            # Record credit transaction
            credit_record = {
                'account_id': account_id,
                'amount': float(amount),
                'currency': currency,
                'type': 'credit',
                'timestamp': datetime.utcnow().isoformat(),
                'new_balance': float(new_balance)
            }
            
            await self.redis.lpush(f"transactions:{account_id}", json.dumps(credit_record))
            
            return {
                'success': True,
                'new_balance': float(new_balance),
                'transaction_id': str(uuid.uuid4())
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _rollback_settlement(self, payment_data: Dict):
        """Rollback settlement in case of failure"""
        try:
            # Reverse debit
            if 'from_account' in payment_data:
                account_key = f"account:{payment_data['from_account']}:{payment_data['currency']}"
                current_balance = await self.redis.get(account_key)
                
                if current_balance is not None:
                    current_balance = Decimal(current_balance)
                    new_balance = current_balance + payment_data['net_amount']
                    await self.redis.set(account_key, str(new_balance))
            
            # Reverse credit
            if 'to_account' in payment_data:
                account_key = f"account:{payment_data['to_account']}:{payment_data['currency']}"
                current_balance = await self.redis.get(account_key)
                
                if current_balance is not None:
                    current_balance = Decimal(current_balance)
                    new_balance = current_balance - payment_data['net_amount']
                    await self.redis.set(account_key, str(new_balance))
            
            logger.info(f"Rollback completed for payment: {payment_data.get('reference', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    async def _get_fx_rate(self, from_currency: str, to_currency: str) -> Dict:
        """Get FX rate with caching and fallback"""
        cache_key = f"fx_rate:{from_currency}:{to_currency}"
        
        # Check cache
        cached_rate = await self.redis.get(cache_key)
        if cached_rate:
            rate_data = json.loads(cached_rate)
            
            # Check if cache is still valid (5 seconds TTL)
            if time.time() - rate_data['timestamp'] < 5:
                return rate_data
        
        # Get rate from provider
        rate_data = await self.fx_provider.get_rate(from_currency, to_currency)
        
        # Cache the rate
        await self.redis.setex(
            cache_key,
            5,  # 5 seconds TTL
            json.dumps(rate_data)
        )
        
        return rate_data
    
    async def _calculate_fees(self, payment_data: Dict) -> Decimal:
        """Calculate fees based on multiple factors"""
        base_fee = Decimal('0.50')
        percentage_fee = Decimal('0.001')  # 0.1%
        
        # Calculate percentage-based fee
        amount = Decimal(str(payment_data['amount']))
        percentage_amount = amount * percentage_fee
        
        # Add priority fee
        priority_fee = Decimal('0')
        if payment_data.get('priority') == 'instant':
            priority_fee = Decimal('1.00')
        elif payment_data.get('priority') == 'priority':
            priority_fee = Decimal('0.50')
        
        # Add quantum security fee
        quantum_fee = Decimal('0.10') if payment_data.get('quantum_secure', False) else Decimal('0')
        
        # Total fee
        total_fee = base_fee + percentage_amount + priority_fee + quantum_fee
        
        # Apply minimum and maximum
        min_fee = Decimal('0.50')
        max_fee = Decimal('50.00')
        
        total_fee = max(min_fee, min(total_fee, max_fee))
        
        return total_fee
    
    async def _validate_and_enrich_payment(self, payment: PaymentRequest, trace_id: str) -> Dict:
        """Validate and enrich payment data"""
        enriched = payment.dict()
        enriched['trace_id'] = trace_id
        enriched['timestamp'] = datetime.utcnow().isoformat()
        
        # Validate account formats
        if not self._validate_account_format(enriched['from_account']):
            raise HTTPException(400, "Invalid source account format")
        
        if not self._validate_account_format(enriched['to_account']):
            raise HTTPException(400, "Invalid destination account format")
        
        # Check for self-transfer
        if enriched['from_account'] == enriched['to_account']:
            raise HTTPException(400, "Cannot transfer to same account")
        
        # Set default settlement currency
        if not enriched.get('settlement_currency'):
            enriched['settlement_currency'] = enriched['currency']
        
        # Add system metadata
        enriched['system_metadata'] = {
            'processing_node': self.config['node_id'],
            'api_version': '2.0.0',
            'quantum_security_level': 5 if enriched['quantum_secure'] else 0
        }
        
        return enriched
    
    def _validate_account_format(self, account: str) -> bool:
        """Validate account format"""
        # Basic validation - in production, this would check against account database
        return len(account) >= 5 and len(account) <= 128
    
    def _group_payments_by_currency(self, payments: List[PaymentRequest]) -> Dict[str, List]:
        """Group payments by currency pair for optimization"""
        groups = {}
        
        for payment in payments:
            from_currency = payment.currency
            to_currency = payment.settlement_currency or payment.currency
            
            key = f"{from_currency}_{to_currency}"
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append(payment)
        
        return groups
    
    async def _process_payment_group(self, currency_pair: str, payments: List[PaymentRequest], batch_id: str) -> Dict:
        """Process a group of payments with same currency pair"""
        group_results = []
        successful = 0
        failed = 0
        total_amount = Decimal('0')
        total_fees = Decimal('0')
        
        # Get FX rate once for the group
        currencies = currency_pair.split('_')
        fx_data = await self._get_fx_rate(currencies[0], currencies[1])
        
        # Process payments in parallel with limits
        semaphore = asyncio.Semaphore(10)  # Limit concurrent processing per group
        
        async def process_single(payment):
            async with semaphore:
                try:
                    # Enrich with group FX rate
                    payment_dict = payment.dict()
                    payment_dict['fx_rate'] = fx_data['rate']
                    payment_dict['fx_timestamp'] = fx_data['timestamp']
                    
                    # Process payment
                    result = await self.process_payment(PaymentRequest(**payment_dict))
                    
                    return {
                        'payment_reference': payment.reference,
                        'status': 'success',
                        'transaction_id': result.transaction_id,
                        'amount': float(payment.amount),
                        'fees': float(result.fees)
                    }
                    
                except Exception as e:
                    return {
                        'payment_reference': payment.reference,
                        'status': 'failed',
                        'error': str(e),
                        'amount': float(payment.amount),
                        'fees': 0
                    }
        
        # Create processing tasks
        tasks = [process_single(payment) for payment in payments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                group_results.append({
                    'status': 'failed',
                    'error': str(result)
                })
                failed += 1
            else:
                group_results.append(result)
                if result['status'] == 'success':
                    successful += 1
                    total_amount += Decimal(str(result['amount']))
                    total_fees += Decimal(str(result['fees']))
                else:
                    failed += 1
        
        return {
            'currency_pair': currency_pair,
            'successful': successful,
            'failed': failed,
            'total_amount': float(total_amount),
            'total_fees': float(total_fees),
            'results': group_results
        }
    
    async def _monitor_performance(self):
        """Monitor and optimize performance"""
        while True:
            try:
                # Collect performance metrics
                metrics = {
                    'queue_size': self.pending_settlements.qsize(),
                    'active_settlements': self.processing_semaphore._value,
                    'redis_connected': await self.redis.ping(),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Update Prometheus metrics
                QUEUE_SIZE.set(metrics['queue_size'])
                
                # Check for performance issues
                if metrics['queue_size'] > 1000:
                    logger.warning(f"High queue size: {metrics['queue_size']}")
                    await self._scale_processing_capacity()
                
                # Log metrics periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    logger.info(f"Performance metrics: {metrics}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _scale_processing_capacity(self):
        """Scale processing capacity based on load"""
        current_capacity = self.processing_semaphore._value
        queue_size = self.pending_settlements.qsize()
        
        # Calculate optimal capacity
        optimal_capacity = min(500, max(100, int(queue_size / 10)))
        
        if optimal_capacity != current_capacity:
            # Update semaphore (simplified - in production would use dynamic scaling)
            logger.info(f"Scaling processing capacity from {current_capacity} to {optimal_capacity}")
            
            # Note: In production, this would scale worker nodes or containers
    
    async def _cleanup_old_data(self):
        """Cleanup old data from cache and temporary storage"""
        while True:
            try:
                # Clean old cache entries
                current_time = time.time()
                cache_keys = await self.redis.keys("cache:*")
                
                for key in cache_keys:
                    ttl = await self.redis.ttl(key)
                    if ttl < -3600:  # Expired more than 1 hour ago
                        await self.redis.delete(key)
                
                # Clean old temporary locks
                lock_keys = await self.redis.keys("lock:*")
                for key in lock_keys:
                    lock_time = await self.redis.get(key)
                    if lock_time and current_time - float(lock_time) > 3600: 
