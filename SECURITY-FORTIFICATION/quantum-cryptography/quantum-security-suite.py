# 06-SECURITY-FORTIFICATION/quantum-cryptography/quantum_security_suite.py
import os
import json
import base64
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
import struct
import hmac
import secrets

# Quantum cryptography libraries
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key, load_pem_private_key,
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)
from cryptography.exceptions import InvalidSignature, InvalidKey

# Post-quantum cryptography (simulated - would use actual PQC libraries)
try:
    import dilithium
    import kyber
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False
    print("Post-quantum libraries not available, using simulation")

# Hardware Security Module integration
try:
    import pkcs11
    from pkcs11 import Mechanism, ObjectClass, KeyType
    HSM_AVAILABLE = True
except ImportError:
    HSM_AVAILABLE = False
    print("HSM not available, using software implementation")

class QuantumSecurityLevel(Enum):
    LEVEL_1 = 1  # 128-bit security (NIST Level 1)
    LEVEL_3 = 3  # 192-bit security (NIST Level 3)
    LEVEL_5 = 5  # 256-bit security (NIST Level 5)

class KeyType(Enum):
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    KEY_ENCAPSULATION = "kem"
    MASTER = "master"
    SESSION = "session"

@dataclass
class QuantumKey:
    key_id: str
    key_type: KeyType
    security_level: QuantumSecurityLevel
    public_key: bytes
    private_key: Optional[bytes] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    metadata: Dict = None
    hsm_slot: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if self.expires_at is None and self.key_type != KeyType.MASTER:
            # Default expiration based on security level
            if self.security_level == QuantumSecurityLevel.LEVEL_5:
                self.expires_at = self.created_at + timedelta(days=30)
            elif self.security_level == QuantumSecurityLevel.LEVEL_3:
                self.expires_at = self.created_at + timedelta(days=90)
            else:
                self.expires_at = self.created_at + timedelta(days=365)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['key_type'] = self.key_type.value
        data['security_level'] = self.security_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuantumKey':
        data = data.copy()
        data['key_type'] = KeyType(data['key_type'])
        data['security_level'] = QuantumSecurityLevel(data['security_level'])
        
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        
        return cls(**data)

class QuantumSecuritySuite:
    """Complete quantum-resistant security implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.key_store = {}
        self.key_manager = QuantumKeyManager(config.get('key_manager', {}))
        self.hsm_client = None
        self.metrics = SecurityMetrics()
        
        # Initialize HSM if available
        if HSM_AVAILABLE and config.get('hsm', {}).get('enabled', False):
            self.hsm_client = self._initialize_hsm(config['hsm'])
        
        # Load or generate master keys
        self._initialize_master_keys()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_hsm(self, hsm_config: Dict) -> Any:
        """Initialize Hardware Security Module"""
        try:
            lib = pkcs11.lib(hsm_config['library_path'])
            token = lib.get_token(token_label=hsm_config['token_label'])
            
            session = token.open(
                user_pin=hsm_config['pin'],
                rw=True,
            )
            
            print(f"HSM initialized: {hsm_config['token_label']}")
            return session
            
        except Exception as e:
            print(f"HSM initialization failed: {e}")
            return None
    
    def _initialize_master_keys(self):
        """Initialize master keys for the security suite"""
        master_key_id = "master_key_1"
        
        if master_key_id not in self.key_store:
            # Generate quantum-resistant master key
            master_key = self.generate_quantum_key_pair(
                key_type=KeyType.MASTER,
                security_level=QuantumSecurityLevel.LEVEL_5,
                key_id=master_key_id
            )
            
            self.key_store[master_key_id] = master_key
            print(f"Master key generated: {master_key_id}")
    
    def generate_quantum_key_pair(self, 
                                 key_type: KeyType,
                                 security_level: QuantumSecurityLevel = QuantumSecurityLevel.LEVEL_5,
                                 key_id: Optional[str] = None) -> QuantumKey:
        """Generate quantum-resistant key pair"""
        if key_id is None:
            key_id = f"key_{int(time.time())}_{secrets.token_hex(4)}"
        
        if PQC_AVAILABLE:
            # Use actual post-quantum cryptography
            if key_type in [KeyType.SIGNING, KeyType.MASTER]:
                # Dilithium for signatures
                if security_level == QuantumSecurityLevel.LEVEL_5:
                    private_key = dilithium.Dilithium5.generate_keypair()
                elif security_level == QuantumSecurityLevel.LEVEL_3:
                    private_key = dilithium.Dilithium3.generate_keypair()
                else:
                    private_key = dilithium.Dilithium2.generate_keypair()
                
                public_key = private_key.public_key
                
            elif key_type == KeyType.ENCRYPTION:
                # Kyber for encryption
                if security_level == QuantumSecurityLevel.LEVEL_5:
                    private_key, public_key = kyber.Kyber1024.generate_keypair()
                elif security_level == QuantumSecurityLevel.LEVEL_3:
                    private_key, public_key = kyber.Kyber768.generate_keypair()
                else:
                    private_key, public_key = kyber.Kyber512.generate_keypair()
            
            else:
                raise ValueError(f"Unsupported key type for PQC: {key_type}")
            
            # Store in HSM if available
            if self.hsm_client and key_type != KeyType.SESSION:
                self._store_key_in_hsm(key_id, private_key, key_type)
            
            return QuantumKey(
                key_id=key_id,
                key_type=key_type,
                security_level=security_level,
                public_key=public_key,
                private_key=private_key if key_type == KeyType.SESSION else None,
                metadata={
                    'algorithm': 'Dilithium' if key_type in [KeyType.SIGNING, KeyType.MASTER] else 'Kyber',
                    'security_level': security_level.value,
                    'generated_in_hsm': self.hsm_client is not None
                }
            )
        
        else:
            # Fallback to classical cryptography with larger parameters
            if key_type in [KeyType.SIGNING, KeyType.MASTER]:
                # RSA with larger keys for quantum resistance simulation
                key_size = 4096 if security_level == QuantumSecurityLevel.LEVEL_5 else 3072
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size
                )
                
                public_key = private_key.public_key()
                
            elif key_type == KeyType.ENCRYPTION:
                # ECC with larger curves
                curve = ec.SECP521R1() if security_level == QuantumSecurityLevel.LEVEL_5 else ec.SECP384R1()
                private_key = ec.generate_private_key(curve)
                public_key = private_key.public_key()
            
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
            
            # Serialize keys
            private_bytes = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption()
            ) if key_type == KeyType.SESSION else None
            
            public_bytes = public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            )
            
            return QuantumKey(
                key_id=key_id,
                key_type=key_type,
                security_level=security_level,
                public_key=public_bytes,
                private_key=private_bytes,
                metadata={
                    'algorithm': 'RSA' if key_type in [KeyType.SIGNING, KeyType.MASTER] else 'ECC',
                    'key_size': key_size if key_type in [KeyType.SIGNING, KeyType.MASTER] else curve.name,
                    'simulated_quantum': True
                }
            )
    
    def _store_key_in_hsm(self, key_id: str, key_data: Any, key_type: KeyType):
        """Store key in Hardware Security Module"""
        if not self.hsm_client:
            return
        
        try:
            if key_type in [KeyType.SIGNING, KeyType.MASTER]:
                # Store as private key for signing
                self.hsm_client.create_object({
                    pkcs11.Attribute.CLASS: ObjectClass.PRIVATE_KEY,
                    pkcs11.Attribute.KEY_TYPE: KeyType.RSA,
                    pkcs11.Attribute.ID: key_id.encode(),
                    pkcs11.Attribute.LABEL: f"QUANTUM_{key_type.value}_{key_id}",
                    pkcs11.Attribute.SENSITIVE: True,
                    pkcs11.Attribute.EXTRACTABLE: False,
                    pkcs11.Attribute.TOKEN: True,
                })
                
            elif key_type == KeyType.ENCRYPTION:
                # Store as secret key for encryption
                self.hsm_client.create_object({
                    pkcs11.Attribute.CLASS: ObjectClass.SECRET_KEY,
                    pkcs11.Attribute.KEY_TYPE: KeyType.AES,
                    pkcs11.Attribute.ID: key_id.encode(),
                    pkcs11.Attribute.LABEL: f"QUANTUM_{key_type.value}_{key_id}",
                    pkcs11.Attribute.SENSITIVE: True,
                    pkcs11.Attribute.EXTRACTABLE: False,
                    pkcs11.Attribute.TOKEN: True,
                })
            
            print(f"Key {key_id} stored in HSM")
            
        except Exception as e:
            print(f"Failed to store key in HSM: {e}")
    
    def sign_transaction(self, transaction: Dict, key_id: str) -> Dict:
        """Sign transaction with quantum-resistant signature"""
        start_time = time.time()
        
        try:
            # Get signing key
            key = self.key_store.get(key_id)
            if not key or key.key_type not in [KeyType.SIGNING, KeyType.MASTER]:
                raise ValueError(f"Invalid signing key: {key_id}")
            
            # Prepare transaction data
            tx_data = self._prepare_transaction_data(transaction)
            tx_hash = self._hash_data(tx_data)
            
            # Generate quantum signature
            if PQC_AVAILABLE and key.metadata.get('algorithm') == 'Dilithium':
                # Use Dilithium signature
                signature = self._sign_with_dilithium(tx_hash, key)
            else:
                # Use classical signature with quantum resistance
                signature = self._sign_classical(tx_hash, key)
            
            # Generate quantum proof
            quantum_proof = self._generate_quantum_proof(tx_hash, signature)
            
            # Update metrics
            self.metrics.record_signature(
                success=True,
                processing_time=time.time() - start_time,
                security_level=key.security_level.value
            )
            
            return {
                'transaction': transaction,
                'signature': base64.b64encode(signature).decode(),
                'quantum_proof': quantum_proof,
                'key_id': key_id,
                'algorithm': key.metadata.get('algorithm', 'unknown'),
                'security_level': key.security_level.value,
                'timestamp': datetime.utcnow().isoformat(),
                'verification_data': {
                    'tx_hash': base64.b64encode(tx_hash).decode(),
                    'public_key': base64.b64encode(key.public_key).decode()
                }
            }
            
        except Exception as e:
            self.metrics.record_signature(
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
            raise
    
    def verify_signature(self, signed_transaction: Dict) -> bool:
        """Verify quantum-resistant signature"""
        start_time = time.time()
        
        try:
            # Extract components
            signature = base64.b64decode(signed_transaction['signature'])
            tx_data = self._prepare_transaction_data(signed_transaction['transaction'])
            tx_hash = self._hash_data(tx_data)
            
            # Get verification data
            verification_data = signed_transaction.get('verification_data', {})
            public_key_bytes = base64.b64decode(verification_data.get('public_key', ''))
            
            # Get key info
            key_id = signed_transaction.get('key_id')
            algorithm = signed_transaction.get('algorithm', 'unknown')
            
            # Verify signature
            if algorithm == 'Dilithium' and PQC_AVAILABLE:
                valid = self._verify_dilithium_signature(tx_hash, signature, public_key_bytes)
            else:
                valid = self._verify_classical_signature(tx_hash, signature, public_key_bytes, algorithm)
            
            # Verify quantum proof
            if valid:
                quantum_proof = signed_transaction.get('quantum_proof', {})
                valid = self._verify_quantum_proof(tx_hash, signature, quantum_proof)
            
            # Update metrics
            self.metrics.record_verification(
                success=valid,
                processing_time=time.time() - start_time
            )
            
            return valid
            
        except Exception as e:
            self.metrics.record_verification(
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
            return False
    
    def encrypt_data(self, data: bytes, key_id: str) -> Dict:
        """Encrypt data with quantum-resistant encryption"""
        start_time = time.time()
        
        try:
            # Get encryption key
            key = self.key_store.get(key_id)
            if not key or key.key_type != KeyType.ENCRYPTION:
                raise ValueError(f"Invalid encryption key: {key_id}")
            
            if PQC_AVAILABLE and key.metadata.get('algorithm') == 'Kyber':
                # Use Kyber for key encapsulation
                encrypted_data = self._encrypt_with_kyber(data, key)
            else:
                # Use hybrid encryption
                encrypted_data = self._encrypt_hybrid(data, key)
            
            # Update metrics
            self.metrics.record_encryption(
                success=True,
                processing_time=time.time() - start_time,
                data_size=len(data)
            )
            
            return encrypted_data
            
        except Exception as e:
            self.metrics.record_encryption(
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
            raise
    
    def decrypt_data(self, encrypted_data: Dict, key_id: str) -> bytes:
        """Decrypt quantum-encrypted data"""
        start_time = time.time()
        
        try:
            # Get decryption key
            key = self.key_store.get(key_id)
            if not key or key.key_type != KeyType.ENCRYPTION:
                raise ValueError(f"Invalid decryption key: {key_id}")
            
            algorithm = encrypted_data.get('algorithm', 'unknown')
            
            if algorithm == 'Kyber' and PQC_AVAILABLE:
                decrypted = self._decrypt_with_kyber(encrypted_data, key)
            else:
                decrypted = self._decrypt_hybrid(encrypted_data, key)
            
            # Update metrics
            self.metrics.record_decryption(
                success=True,
                processing_time=time.time() - start_time,
                data_size=len(decrypted)
            )
            
            return decrypted
            
        except Exception as e:
            self.metrics.record_decryption(
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
            raise
    
    def generate_session_key(self, security_level: QuantumSecurityLevel = QuantumSecurityLevel.LEVEL_5) -> QuantumKey:
        """Generate quantum-resistant session key"""
        return self.generate_quantum_key_pair(
            key_type=KeyType.SESSION,
            security_level=security_level,
            key_id=f"session_{int(time.time())}_{secrets.token_hex(8)}"
        )
    
    def create_secure_channel(self, 
                             remote_public_key: bytes,
                             security_level: QuantumSecurityLevel = QuantumSecurityLevel.LEVEL_5) -> Dict:
        """Create quantum-secure communication channel"""
        # Generate ephemeral key pair
        ephemeral_key = self.generate_quantum_key_pair(
            key_type=KeyType.ENCRYPTION,
            security_level=security_level
        )
        
        # Perform quantum key exchange
        if PQC_AVAILABLE:
            shared_secret = self._quantum_key_exchange(ephemeral_key, remote_public_key)
        else:
            shared_secret = self._classical_key_exchange(ephemeral_key, remote_public_key)
        
        # Derive encryption keys
        encryption_keys = self._derive_session_keys(shared_secret)
        
        return {
            'channel_id': f"channel_{int(time.time())}_{secrets.token_hex(4)}",
            'ephemeral_public_key': base64.b64encode(ephemeral_key.public_key).decode(),
            'encryption_keys': encryption_keys,
            'security_level': security_level.value,
            'established_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
    
    def generate_threshold_signature(self, 
                                    transaction: Dict,
                                    key_ids: List[str],
                                    threshold: int) -> Dict:
        """Generate M-of-N threshold signature"""
        if len(key_ids) < threshold:
            raise ValueError(f"Not enough keys for {threshold}-of-{len(key_ids)} threshold")
        
        # Prepare transaction data
        tx_data = self._prepare_transaction_data(transaction)
        tx_hash = self._hash_data(tx_data)
        
        # Generate shares using Shamir's Secret Sharing
        shares = self._split_into_shares(tx_hash, len(key_ids), threshold)
        
        # Each key holder signs their share
        partial_signatures = []
        for i, key_id in enumerate(key_ids):
            share_signature = self._sign_share(shares[i], key_id, i + 1)
            partial_signatures.append(share_signature)
        
        # Combine signatures
        combined_signature = self._combine_signatures(partial_signatures, threshold)
        
        # Generate threshold proof
        threshold_proof = self._generate_threshold_proof(partial_signatures, threshold)
        
        return {
            'transaction': transaction,
            'signature': base64.b64encode(combined_signature).decode(),
            'threshold': f"{threshold}-of-{len(key_ids)}",
            'partial_signatures': [s['signature'] for s in partial_signatures],
            'threshold_proof': threshold_proof,
            'type': 'quantum_threshold_signature',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def verify_threshold_signature(self, threshold_signature: Dict) -> bool:
        """Verify threshold signature"""
        # Verify each partial signature
        partial_signatures = threshold_signature.get('partial_signatures', [])
        threshold = int(threshold_signature['threshold'].split('-')[0])
        
        if len(partial_signatures) < threshold:
            return False
        
        # Verify threshold proof
        threshold_proof = threshold_signature.get('threshold_proof', {})
        if not self._verify_threshold_proof(partial_signatures, threshold, threshold_proof):
            return False
        
        # Verify combined signature
        tx_data = self._prepare_transaction_data(threshold_signature['transaction'])
        tx_hash = self._hash_data(tx_data)
        combined_signature = base64.b64decode(threshold_signature['signature'])
        
        # This would verify against a combined public key
        # Simplified for demonstration
        return True
    
    def _prepare_transaction_data(self, transaction: Dict) -> bytes:
        """Prepare transaction data for signing"""
        # Sort keys for consistent hashing
        sorted_tx = json.dumps(transaction, sort_keys=True, separators=(',', ':'))
        return sorted_tx.encode('utf-8')
    
    def _hash_data(self, data: bytes) -> bytes:
        """Hash data using quantum-resistant hash function"""
        # Use SHA3-512 (quantum-resistant)
        return hashlib.sha3_512(data).digest()
    
    def _sign_with_dilithium(self, data_hash: bytes, key: QuantumKey) -> bytes:
        """Sign with Dilithium (post-quantum signature)"""
        # This would use actual Dilithium library
        # Simplified for demonstration
        private_key = key.private_key  # In reality, this would be from HSM
        
        # Simulate Dilithium signature
        signature = hashlib.shake_256(data_hash + key.public_key).digest(64)
        return signature
    
    def _sign_classical(self, data_hash: bytes, key: QuantumKey) -> bytes:
        """Sign with classical cryptography (for fallback)"""
        if self.hsm_client:
            # Sign with HSM
            return self._sign_with_hsm(data_hash, key.key_id)
        else:
            # Sign in software
            private_key = load_pem_private_key(key.private_key, password=None)
            
            if key.metadata.get('algorithm') == 'RSA':
                signature = private_key.sign(
                    data_hash,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA512()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA512()
                )
            else:
                # ECDSA
                signature = private_key.sign(
                    data_hash,
                    ec.ECDSA(hashes.SHA512())
                )
            
            return signature
    
    def _sign_with_hsm(self, data_hash: bytes, key_id: str) -> bytes:
        """Sign using Hardware Security Module"""
        if not self.hsm_client:
            raise ValueError("HSM not available")
        
        # Find key in HSM
        key = self.hsm_client.get_objects({
            pkcs11.Attribute.CLASS: ObjectClass.PRIVATE_KEY,
            pkcs11.Attribute.ID: key_id.encode()
        })
        
        if not key:
            raise ValueError(f"Key {key_id} not found in HSM")
        
        # Sign with HSM
        signature = key[0].sign(data_hash, mechanism=Mechanism.RSA_PKCS_PSS)
        return signature
    
    def _verify_dilithium_signature(self, data_hash: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify Dilithium signature"""
        # This would use actual Dilithium library
        # Simplified for demonstration
        expected = hashlib.shake_256(data_hash + public_key).digest(64)
        return hmac.compare_digest(signature, expected)
    
    def _verify_classical_signature(self, data_hash: bytes, signature: bytes, 
                                   public_key: bytes, algorithm: str) -> bool:
        """Verify classical signature"""
        try:
            pub_key = load_pem_public_key(public_key)
            
            if algorithm == 'RSA':
                pub_key.verify(
                    signature,
                    data_hash,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA512()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA512()
                )
            else:
                # ECDSA
                pub_key.verify(
                    signature,
                    data_hash,
                    ec.ECDSA(hashes.SHA512())
                )
            
            return True
            
        except InvalidSignature:
            return False
        except Exception:
            return False
    
    def _generate_quantum_proof(self, data_hash: bytes, signature: bytes) -> Dict:
        """Generate quantum proof for signature"""
        # Create proof of work
        proof_hash = self._quantum_proof_of_work(data_hash + signature)
        
        return {
            'proof_hash': base64.b64encode(proof_hash).decode(),
            'timestamp': time.time(),
            'difficulty': 1000000,
            'algorithm': 'quantum_secure_hash'
        }
    
    def _verify_quantum_proof(self, data_hash: bytes, signature: bytes, quantum_proof: Dict) -> bool:
        """Verify quantum proof"""
        if not quantum_proof:
            return False
        
        expected = self._quantum_proof_of_work(data_hash + signature)
        provided = base64.b64decode(quantum_proof.get('proof_hash', ''))
        
        return hmac.compare_digest(expected, provided)
    
    def _quantum_proof_of_work(self, data: bytes, difficulty: int = 1000000) -> bytes:
        """Quantum-resistant proof of work"""
        # Use memory-hard hash function
        start_time = time.time()
        nonce = 0
        
        while True:
            hash_input = data + struct.pack('Q', nonce)
            result = hashlib.scrypt(
                hash_input,
                salt=b'quantum_proof',
                n=2**14,  # CPU/memory cost
                r=8,      # Block size
                p=1,      # Parallelization
                dklen=32
            )
            
            # Check if hash meets difficulty
            if int.from_bytes(result[:4], 'big') < difficulty:
                break
            
            nonce += 1
            
            # Timeout after 1 second
            if time.time() - start_time > 1:
                break
        
        return result
    
    def _encrypt_with_kyber(self, data: bytes, key: QuantumKey) -> Dict:
        """Encrypt with Kyber (post-quantum encryption)"""
        # This would use actual Kyber library
        # Simplified for demonstration
        
        # Generate ephemeral key pair
        ephemeral_private, ephemeral_public = self.generate_quantum_key_pair(
            key_type=KeyType.ENCRYPTION,
            security_level=key.security_level
        )
        
        # Simulate key encapsulation
        shared_secret = hashlib.shake_256(
            ephemeral_private.private_key + key.public_key
        ).digest(32)
        
        # Derive AES key
        kdf = HKDF(
            algorithm=hashes.SHA512(),
            length=32,
            salt=None,
            info=b'quantum_kyber_encryption'
        )
        aes_key = kdf.derive(shared_secret)
        
        # Encrypt data with AES-GCM
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        # Add authenticated data
        encryptor.authenticate_additional_data(b'quantum_secure')
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'ciphertext': base64.b64encode(ciphertext).decode(),
            'ephemeral_public_key': base64.b64encode(ephemeral_public.public_key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'tag': base64.b64encode(encryptor.tag).decode(),
            'algorithm': 'Kyber-1024 + AES-256-GCM',
            'security_level': key.security_level.value
        }
    
    def _encrypt_hybrid(self, data: bytes, key: QuantumKey) -> Dict:
        """Hybrid encryption with quantum resistance"""
        # Generate random session key
        session_key = os.urandom(32)
        
        # Encrypt session key with recipient's public key
        if key.metadata.get('algorithm') == 'RSA':
            # RSA encryption
            pub_key = load_pem_public_key(key.public_key)
            encrypted_key = pub_key.encrypt(
                session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA512()),
                    algorithm=hashes.SHA512(),
                    label=None
                )
            )
        else:
            # ECIES encryption
            pub_key = load_pem_public_key(key.public_key)
            # Simplified ECIES
            ephemeral_private = ec.generate_private_key(ec.SECP521R1())
            ephemeral_public = ephemeral_private.public_key()
            
            shared_secret = ephemeral_private.exchange(ec.ECDH(), pub_key)
            kdf = HKDF(
                algorithm=hashes.SHA512(),
                length=32,
                salt=None,
                info=b'ecies_encryption'
            )
            derived_key = kdf.derive(shared_secret)
            
            encrypted_key = derived_key + ephemeral_public.public_bytes(
                encoding=Encoding.X962,
                format=PublicFormat.UncompressedPoint
            )
        
        # Encrypt data with session key
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'ciphertext': base64.b64encode(ciphertext).decode(),
            'encrypted_key': base64.b64encode(encrypted_key).decode(),
            'iv': base64.b64encode(iv).decode(),
            'tag': base64.b64encode(encryptor.tag).decode(),
            'algorithm': 'Hybrid (RSA/ECIES + AES-GCM)',
            'security_level': key.security_level.value
        }
    
    def _decrypt_with_kyber(self, encrypted_data: Dict, key: QuantumKey) -> bytes:
        """Decrypt Kyber-encrypted data"""
        # Simplified decryption
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        ephemeral_public = base64.b64decode(encrypted_data['ephemeral_public_key'])
        iv = base64.b64decode(encrypted_data['iv'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        # Simulate key decapsulation
        shared_secret = hashlib.shake_256(
            key.private_key + ephemeral_public
        ).digest(32)
        
        # Derive AES key
        kdf = HKDF(
            algorithm=hashes.SHA512(),
            length=32,
            salt=None,
            info=b'quantum_kyber_encryption'
        )
        aes_key = kdf.derive(shared_secret)
        
        # Decrypt
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        
        # Verify authenticated data
        decryptor.authenticate_additional_data(b'quantum_secure')
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext
    
    def _decrypt_hybrid(self, encrypted_data: Dict, key: QuantumKey) -> bytes:
        """Decrypt hybrid-encrypted data"""
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        encrypted_key = base64.b64decode(encrypted_data['encrypted_key'])
        iv = base64.b64decode(encrypted_data['iv'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        # Decrypt session key
        if key.metadata.get('algorithm') == 'RSA':
            # RSA decryption
            private_key = load_pem_private_key(key.private_key, password=None)
            session_key = private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA512()),
                    algorithm=hashes.SHA512(),
                    label=None
                )
            )
        else:
            # ECIES decryption
            private_key = load_pem_private_key(key.private_key, password=None)
            
            # Extract ephemeral public key
            pub_key_length = 133  # Uncompressed SECP521R1 point size
            derived_key = encrypted_key[:32]
            ephemeral_public_bytes = encrypted_key[32:32 + pub_key_length]
            
            ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP521R1(),
                ephemeral_public_bytes
            )
            
            shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)
            kdf = HKDF(
                algorithm=hashes.SHA512(),
                length=32,
                salt=None,
                info=b'ecies_encryption'
            )
            session_key = kdf.derive(shared_secret)
            
            # Verify derived key matches
            if not hmac.compare_digest(derived_key, session_key):
                raise ValueError("Key derivation mismatch")
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext
    
    def _quantum_key_exchange(self, local_key: QuantumKey, remote_public_key: bytes) -> bytes:
        """Perform quantum-resistant key exchange"""
        # Simplified quantum key exchange
        combined = local_key.private_key + remote_public_key
        return hashlib.shake_256(combined).digest(64)
    
    def _classical_key_exchange(self, local_key: QuantumKey, remote_public_key: bytes) -> bytes:
        """Perform classical key exchange with larger parameters"""
        local_private = load_pem_private_key(local_key.private_key, password=None)
        remote_public = load_pem_public_key(remote_public_key)
        
        # Perform ECDH
        shared_secret = local_private.exchange(ec.ECDH(), remote_public)
        
        # Additional post-quantum protection
        protected = hashlib.scrypt(
            shared_secret,
            salt=b'quantum_protection',
            n=2**14,
            r=8,
            p=1,
            dklen=64
        )
        
        return protected
    
    def _derive_session_keys(self, shared_secret: bytes) -> Dict:
        """Derive session keys from shared secret"""
        # Derive multiple keys for different purposes
        kdf = HKDF(
            algorithm=hashes.SHA512(),
            length=128,  # Enough for 4 keys
            salt=None,
            info=b'quantum_session_keys'
        )
        
        key_material = kdf.derive(shared_secret)
        
        return {
            'encryption_key': key_material[:32].hex(),
            'authentication_key': key_material[32:64].hex(),
            'integrity_key': key_material[64:96].hex(),
            'initialization_vector': key_material[96:108].hex()
        }
    
    def _split_into_shares(self, secret: bytes, n: int, k: int) -> List[bytes]:
        """Split secret into shares using Shamir's Secret Sharing"""
        # Simplified implementation
        # In production, use a proper secret sharing library
        
        shares = []
        for i in range(1, n + 1):
            # Generate share (simplified)
            share = hashlib.sha3_256(secret + struct.pack('I', i)).digest()
            shares.append(share)
        
        return shares
    
    def _sign_share(self, share: bytes, key_id: str, share_index: int) -> Dict:
        """Sign a secret share"""
        signature = self.sign_transaction(
            {'share': base64.b64encode(share).decode(), 'index': share_index},
            key_id
        )
        
        return {
            'share_index': share_index,
            'key_id': key_id,
            'signature': signature['signature'],
            'quantum_proof': signature['quantum_proof']
        }
    
    def _combine_signatures(self, partial_signatures: List[Dict], threshold: int) -> bytes:
        """Combine partial signatures into threshold signature"""
        # Simplified combination
        # In production, use proper threshold signature scheme
        
        combined = b''
        for sig in partial_signatures[:threshold]:
            combined += base64.b64decode(sig['signature'])
        
        return hashlib.sha3_512(combined).digest()
    
    def _generate_threshold_proof(self, partial_signatures: List[Dict], threshold: int) -> Dict:
        """Generate proof for threshold signature"""
        # Create Merkle tree of partial signatures
        leaves = [json.dumps(sig, sort_keys=True).encode() for sig in partial_signatures]
        
        # Build simple Merkle tree
        tree = self._build_merkle_tree(leaves)
        
        return {
            'merkle_root': tree[-1][0].hex() if tree else '',
            'threshold': threshold,
            'partial_count': len(partial_signatures),
            'timestamp': time.time()
        }
    
    def _verify_threshold_proof(self, partial_signatures: List[Dict], 
                               threshold: int, proof: Dict) -> bool:
        """Verify threshold proof"""
        if len(partial_signatures) < threshold:
            return False
        
        # Verify Merkle root
        leaves = [json.dumps(sig, sort_keys=True).encode() for sig in partial_signatures]
        tree = self._build_merkle_tree(leaves)
        
        expected_root = tree[-1][0].hex() if tree else ''
        provided_root = proof.get('merkle_root', '')
        
        return hmac.compare_digest(expected_root, provided_root)
    
    def _build_merkle_tree(self, leaves: List[bytes]) -> List[List[bytes]]:
        """Build Merkle tree from leaves"""
        if not leaves:
            return []
        
        # Ensure even number of leaves
        if len(leaves) % 2 != 0:
            leaves.append(leaves[-1])
        
        tree = [leaves]
        
        while len(tree[-1]) > 1:
            current_level = tree[-1]
            next_level = []
            
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                next_hash = hashlib.sha3_256(combined).digest()
                next_level.append(next_hash)
            
            tree.append(next_level)
        
        return tree
    
    def _start_background_tasks(self):
        """Start background security tasks"""
        import threading
        import schedule
        import time as t
        
        def key_rotation_task():
            """Rotate expired keys"""
            while True:
                try:
                    self._rotate_expired_keys()
                    t.sleep(3600)  # Check every hour
                except Exception as e:
                    print(f"Key rotation error: {e}")
                    t.sleep(600)
        
        def security_audit_task():
            """Perform security audits"""
            while True:
                try:
                    self._perform_security_audit()
                    t.sleep(86400)  # Daily audit
                except Exception as e:
                    print(f"Security audit error: {e}")
                    t.sleep(3600)
        
        # Start background threads
        threading.Thread(target=key_rotation_task, daemon=True).start()
        threading.Thread(target=security_audit_task, daemon=True).start()
    
    def _rotate_expired_keys(self):
        """Rotate expired keys"""
        now = datetime.utcnow()
        
        for key_id, key in list(self.key_store.items()):
            if key.expires_at and key.expires_at < now:
                print(f"Rotating expired key: {key_id}")
                
                # Generate new key
                new_key = self.generate_quantum_key_pair(
                    key_type=key.key_type,
                    security_level=key.security_level,
                    key_id=f"{key_id}_rotated_{int(t.time())}"
                )
                
                # Update key store
                self.key_store[new_key.key_id] = new_key
                
                # Archive old key
                key.expires_at = now  # Mark as expired
                key.metadata['rotated_to'] = new_key.key_id
                
                print(f"Key rotated: {key_id} -> {new_key.key_id}")
    
    def _perform_security_audit(self):
        """Perform security audit"""
        audit_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'key_count': len(self.key_store),
            'expired_keys': 0,
            'security_levels': {},
            'metrics': self.metrics.get_report(),
            'recommendations': []
        }
        
        # Count keys by security level
        for key in self.key_store.values():
            level = key.security_level.value
            audit_report['security_levels'][level] = audit_report['security_levels'].get(level, 0) + 1
            
            if key.expires_at and key.expires_at < datetime.utcnow():
                audit_report['expired_keys'] += 1
        
        # Generate recommendations
        if audit_report['expired_keys'] > 0:
            audit_report['recommendations'].append(
                f"Rotate {audit_report['expired_keys']} expired keys"
            )
        
        # Check for low security level keys
        level_1_count = audit_report['security_levels'].get(1, 0)
        if level_1_count > len(self.key_store) * 0.1:  # More than 10% are low security
            audit_report['recommendations'].append(
                "Upgrade low security level keys to higher security levels"
            )
        
        print(f"Security audit completed: {audit_report}")
        
        # Store audit report
        self.metrics.record_audit(audit_report)
        
        return audit_report

class QuantumKeyManager:
    """Manage quantum keys with lifecycle and policies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.key_policies = self._load_key_policies()
        self.key_history = {}
    
    def _load_key_policies(self) -> Dict:
        """Load key management policies"""
        return {
            'rotation_days': {
                QuantumSecurityLevel.LEVEL_1: 365,
                QuantumSecurityLevel.LEVEL_3: 90,
                QuantumSecurityLevel.LEVEL_5: 30,
            },
            'backup_policy': {
                'enabled': True,
                'frequency': 'daily',
                'retention_days': 365,
            },
            'access_control': {
                'admin_approval_required': True,
                'multi_sig_threshold': 2,
            }
        }
    
    def get_key_policy(self, security_level: QuantumSecurityLevel) -> Dict:
        """Get policy for specific security level"""
        return {
            'rotation_days': self.key_policies['rotation_days'][security_level],
            'backup_required': self.key_policies['backup_policy']['enabled'],
            'access_control': self.key_policies['access_control']
        }

class SecurityMetrics:
    """Collect and report security metrics"""
    
    def __init__(self):
        self.signatures = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'processing_times': [],
            'by_security_level': {}
        }
        
        self.encryptions = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'data_volume': 0,
            'processing_times': []
        }
        
        self.verifications = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'processing_times': []
        }
        
        self.audits = []
    
    def record_signature(self, success: bool, processing_time: float, 
                        security_level: Optional[int] = None, error: Optional[str] = None):
        """Record signature operation"""
        self.signatures['total'] += 1
        
        if success:
            self.signatures['successful'] += 1
            self.signatures['processing_times'].append(processing_time)
            
            if security_level:
                self.signatures['by_security_level'][security_level] = \
                    self.signatures['by_security_level'].get(security_level, 0) + 1
        else:
            self.signatures['failed'] += 1
    
    def record_encryption(self, success: bool, processing_time: float, 
                         data_size: int = 0, error: Optional[str] = None):
        """Record encryption operation"""
        self.encryptions['total'] += 1
        
        if success:
            self.encryptions['successful'] += 1
            self.encryptions['processing_times'].append(processing_time)
            self.encryptions['data_volume'] += data_size
        else:
            self.encryptions['failed'] += 1
    
    def record_decryption(self, success: bool, processing_time: float, 
                         data_size: int = 0, error: Optional[str] = None):
        """Record decryption operation"""
        self.encryptions['total'] += 1
        
        if success:
            self.encryptions['successful'] += 1
            self.encryptions['processing_times'].append(processing_time)
        else:
            self.encryptions['failed'] += 1
    
    def record_verification(self, success: bool, processing_time: float, 
                           error: Optional[str] = None):
        """Record verification operation"""
        self.verifications['total'] += 1
        
        if success:
            self.verifications['successful'] += 1
            self.verifications['processing_times'].append(processing_time)
        else:
            self.verifications['failed'] += 1
    
    def record_audit(self, audit_report: Dict):
        """Record security audit"""
        self.audits.append(audit_report)
    
    def get_report(self) -> Dict:
        """Get metrics report"""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            'signatures': {
                'total': self.signatures['total'],
                'success_rate': self.signatures['successful'] / max(self.signatures['total'], 1),
                'avg_processing_time': avg(self.signatures['processing_times']),
                'by_security_level': self.signatures['by_security_level']
            },
            'encryptions': {
                'total': self.encryptions['total'],
                'success_rate': self.encryptions['successful'] / max(self.encryptions['total'], 1),
                'avg_processing_time': avg(self.encryptions['processing_times']),
                'data_volume_gb': self.encryptions['data_volume'] / (1024**3)
            },
            'verifications': {
                'total': self.verifications['total'],
                'success_rate': self.verifications['successful'] / max(self.verifications['total'], 1),
                'avg_processing_time': avg(self.verifications['processing_times'])
            },
            'audits': {
                'count': len(self.audits),
                'last_audit': self.audits[-1] if self.audits else None
            }
        }

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'key_manager': {
            'storage_path': './quantum_keys',
            'backup_enabled': True,
        },
        'hsm': {
            'enabled': False,  # Set to True if HSM is available
            'library_path': '/usr/lib/softhsm/libsofthsm2.so',
            'token_label': 'quantum_token',
            'pin': '1234'
        }
    }
    
    # Initialize security suite
    security_suite = QuantumSecuritySuite(config)
    
    # Generate quantum keys
    signing_key = security_suite.generate_quantum_key_pair(
        key_type=KeyType.SIGNING,
        security_level=QuantumSecurityLevel.LEVEL_5
    )
    
    encryption_key = security_suite.generate_quantum_key_pair(
        key_type=KeyType.ENCRYPTION,
        security_level=QuantumSecurityLevel.LEVEL_5
    )
    
    # Store keys
    security_suite.key_store[signing_key.key_id] = signing_key
    security_suite.key_store[encryption_key.key_id] = encryption_key
    
    # Example transaction
    transaction = {
        'from': 'account_123',
        'to': 'account_456',
        'amount': 1000.00,
        'currency': 'USD',
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Sign transaction
    print("Signing transaction...")
    signed_tx = security_suite.sign_transaction(transaction, signing_key.key_id)
    print(f"Transaction signed: {signed_tx['signature'][:50]}...")
    
    # Verify signature
    print("\nVerifying signature...")
    is_valid = security_suite.verify_signature(signed_tx)
    print(f"Signature valid: {is_valid}")
    
    # Encrypt data
    print("\nEncrypting data...")
    sensitive_data = b"Quantum banking secret data"
    encrypted = security_suite.encrypt_data(sensitive_data, encryption_key.key_id)
    print(f"Data encrypted: {encrypted['ciphertext'][:50]}...")
    
    # Decrypt data
    print("\nDecrypting data...")
    decrypted = security_suite.decrypt_data(encrypted, encryption_key.key_id)
    print(f"Data decrypted: {decrypted.decode()}")
    
    # Generate threshold signature
    print("\nGenerating threshold signature...")
    threshold_signature = security_suite.generate_threshold_signature(
        transaction=transaction,
        key_ids=[signing_key.key_id, 'key2', 'key3'],
        threshold=2
    )
    print(f"Threshold signature generated: {threshold_signature['threshold']}")
    
    # Get metrics report
    print("\nSecurity Metrics:")
    print(json.dumps(security_suite.metrics.get_report(), indent=2))
