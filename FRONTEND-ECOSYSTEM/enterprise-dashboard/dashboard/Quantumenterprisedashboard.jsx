// 04-FRONTEND-ECOSYSTEM/enterprise-dashboard/dashboard/QuantumEnterpriseDashboard.jsx
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { 
  BrowserRouter as Router, 
  Routes, 
  Route, 
  Navigate,
  useLocation,
  useNavigate 
} from 'react-router-dom';
import { 
  ThemeProvider, 
  createTheme, 
  CssBaseline, 
  GlobalStyles 
} from '@mui/material';
import { 
  Alert, 
  Snackbar, 
  Backdrop, 
  CircularProgress,
  Fade,
  Zoom,
  Slide 
} from '@mui/material';
import { 
  Analytics, 
  Security, 
  AccountBalance, 
  Payment, 
  TrendingUp,
  ShowChart,
  AccountCircle,
  Notifications,
  Settings,
  Dashboard as DashboardIcon,
  SwapHoriz,
  AccountBalanceWallet,
  Timeline,
  BarChart,
  PieChart,
  Cloud,
  Lock,
  Speed,
  FiberSmartRecord,
  QuantumSimulation
} from '@mui/icons-material';
import { 
  motion, 
  AnimatePresence,
  useAnimation,
  useMotionValue,
  useTransform
} from 'framer-motion';
import * as d3 from 'd3';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  BarElement,
  ArcElement,
  Title, 
  Tooltip, 
  Legend,
  Filler,
  RadialLinearScale
} from 'chart.js';
import { 
  Line, 
  Bar, 
  Pie, 
  Doughnut, 
  Radar, 
  PolarArea,
  Scatter,
  Bubble 
} from 'react-chartjs-2';
import { 
  WebSocketProvider, 
  useWebSocket 
} from '../hooks/useWebSocket';
import { 
  useQuantumBanking, 
  QuantumBankingProvider 
} from '../contexts/QuantumBankingContext';
import { 
  useQuantumSecurity, 
  QuantumSecurityProvider 
} from '../contexts/QuantumSecurityContext';
import { 
  useMarketData, 
  MarketDataProvider 
} from '../contexts/MarketDataContext';
import { 
  useRiskManagement, 
  RiskManagementProvider 
} from '../contexts/RiskManagementContext';
import { 
  useCompliance, 
  ComplianceProvider 
} from '../contexts/ComplianceContext';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  RadialLinearScale
);

// Quantum Theme
const quantumTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00e5ff',
      light: '#69f0ff',
      dark: '#00b2cc',
    },
    secondary: {
      main: '#7c4dff',
      light: '#b47cff',
      dark: '#3f1dcb',
    },
    background: {
      default: '#0a0e17',
      paper: '#13182c',
    },
    success: {
      main: '#00ff88',
      light: '#5dffb0',
      dark: '#00ca61',
    },
    warning: {
      main: '#ff9100',
      light: '#ffb74d',
      dark: '#c56200',
    },
    error: {
      main: '#ff1744',
      light: '#ff616f',
      dark: '#c4001d',
    },
    info: {
      main: '#2979ff',
      light: '#75a7ff',
      dark: '#004ecb',
    },
    quantum: {
      main: '#9c27b0',
      light: '#d05ce3',
      dark: '#6a0080',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      background: 'linear-gradient(45deg, #00e5ff 30%, #7c4dff 90%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontWeight: 700,
    },
    h3: {
      fontWeight: 600,
    },
    h4: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 600,
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 25px rgba(0, 229, 255, 0.3)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          background: 'linear-gradient(145deg, #13182c 0%, #0f1424 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(124, 77, 255, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: 'rgba(124, 77, 255, 0.3)',
            boxShadow: '0 12px 48px rgba(124, 77, 255, 0.2)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

// Global Styles
const globalStyles = {
  body: {
    margin: 0,
    padding: 0,
    overflowX: 'hidden',
  },
  '::-webkit-scrollbar': {
    width: '10px',
    height: '10px',
  },
  '::-webkit-scrollbar-track': {
    background: '#0a0e17',
  },
  '::-webkit-scrollbar-thumb': {
    background: 'linear-gradient(45deg, #00e5ff, #7c4dff)',
    borderRadius: '5px',
  },
  '::-webkit-scrollbar-thumb:hover': {
    background: 'linear-gradient(45deg, #7c4dff, #00e5ff)',
  },
};

// Main Dashboard Component
const QuantumEnterpriseDashboard = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [systemStatus, setSystemStatus] = useState('normal');
  const [quantumState, setQuantumState] = useState('active');
  const [activeView, setActiveView] = useState('overview');
  
  const { user, accounts, transactions, fxRates, refreshData } = useQuantumBanking();
  const { securityStatus, quantumProofs, threats } = useQuantumSecurity();
  const { marketData, arbitrageOpportunities, subscribeToMarket } = useMarketData();
  const { riskMetrics, riskAlerts } = useRiskManagement();
  const { complianceStatus, complianceAlerts } = useCompliance();
  
  const ws = useWebSocket('wss://api.quantumbank.ai/ws/dashboard', {
    onMessage: handleWebSocketMessage,
    onReconnect: handleReconnect,
  });
  
  // Animation controls
  const controls = useAnimation();
  const quantumPulse = useMotionValue(0);
  const securityGlow = useTransform(quantumPulse, [0, 1], ['#00e5ff', '#ff1744']);
  
  // Refs for charts
  const quantumChartRef = useRef();
  const marketChartRef = useRef();
  const riskChartRef = useRef();
  
  // Handle WebSocket messages
  function handleWebSocketMessage(data) {
    switch (data.type) {
      case 'transaction_update':
        handleTransactionUpdate(data.payload);
        break;
      case 'market_update':
        handleMarketUpdate(data.payload);
        break;
      case 'security_alert':
        handleSecurityAlert(data.payload);
        break;
      case 'compliance_update':
        handleComplianceUpdate(data.payload);
        break;
      case 'risk_alert':
        handleRiskAlert(data.payload);
        break;
      case 'system_status':
        handleSystemStatus(data.payload);
        break;
      case 'quantum_state':
        handleQuantumState(data.payload);
        break;
    }
  }
  
  // Handle reconnection
  function handleReconnect() {
    console.log('WebSocket reconnected');
    refreshData();
  }
  
  // Handle transaction updates
  function handleTransactionUpdate(transaction) {
    setNotifications(prev => [
      {
        id: Date.now(),
        type: 'transaction',
        title: 'New Transaction',
        message: `$${transaction.amount} ${transaction.currency} settlement`,
        timestamp: new Date(),
        data: transaction,
      },
      ...prev.slice(0, 9)
    ]);
  }
  
  // Handle market updates
  function handleMarketUpdate(marketData) {
    // Update market charts
    if (marketChartRef.current) {
      // Update chart data
    }
  }
  
  // Handle security alerts
  function handleSecurityAlert(alert) {
    setAlerts(prev => [
      {
        id: Date.now(),
        type: 'security',
        severity: alert.severity,
        title: 'Security Alert',
        message: alert.message,
        timestamp: new Date(),
        data: alert,
      },
      ...prev.slice(0, 4)
    ]);
    
    // Pulse quantum indicator
    controls.start({
      scale: [1, 1.2, 1],
      transition: { duration: 0.5 }
    });
  }
  
  // Handle compliance updates
  function handleComplianceUpdate(update) {
    setNotifications(prev => [
      {
        id: Date.now(),
        type: 'compliance',
        title: 'Compliance Update',
        message: update.message,
        timestamp: new Date(),
        data: update,
      },
      ...prev.slice(0, 9)
    ]);
  }
  
  // Handle risk alerts
  function handleRiskAlert(alert) {
    setAlerts(prev => [
      {
        id: Date.now(),
        type: 'risk',
        severity: alert.severity,
        title: 'Risk Alert',
        message: alert.message,
        timestamp: new Date(),
        data: alert,
      },
      ...prev.slice(0, 4)
    ]);
  }
  
  // Handle system status
  function handleSystemStatus(status) {
    setSystemStatus(status.level);
  }
  
  // Handle quantum state
  function handleQuantumState(state) {
    setQuantumState(state.status);
  }
  
  // Calculate dashboard metrics
  const dashboardMetrics = useMemo(() => {
    const totalAssets = Object.values(accounts || {}).reduce(
      (sum, acc) => sum + (acc.balance || 0), 0
    );
    
    const dailyVolume = (transactions || []).reduce(
      (sum, tx) => sum + (tx.amount || 0), 0
    );
    
    const settlementSpeed = calculateAverageSettlementSpeed(transactions);
    const securityScore = calculateSecurityScore(alerts, securityStatus);
    const riskScore = riskMetrics?.overallScore || 0;
    const complianceScore = complianceStatus?.score || 0;
    
    return {
      totalAssets,
      dailyVolume,
      settlementSpeed,
      securityScore,
      riskScore,
      complianceScore,
      quantumEfficiency: calculateQuantumEfficiency(transactions),
      systemUptime: calculateSystemUptime(),
      transactionSuccessRate: calculateSuccessRate(transactions),
    };
  }, [accounts, transactions, alerts, securityStatus, riskMetrics, complianceStatus]);
  
  // Quantum visualization data
  const quantumChartData = useMemo(() => {
    return {
      labels: ['Qubit 1', 'Qubit 2', 'Qubit 3', 'Qubit 4', 'Qubit 5'],
      datasets: [
        {
          label: 'Quantum State',
          data: [0.7, 0.9, 0.5, 0.8, 0.6],
          backgroundColor: 'rgba(0, 229, 255, 0.2)',
          borderColor: '#00e5ff',
          borderWidth: 2,
          tension: 0.4,
          fill: true,
        },
      ],
    };
  }, []);
  
  // Market data chart
  const marketChartData = useMemo(() => {
    return {
      labels: marketData?.labels || ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
      datasets: [
        {
          label: 'FX Volume',
          data: marketData?.volume || [65, 59, 80, 81, 56, 55],
          borderColor: '#7c4dff',
          backgroundColor: 'rgba(124, 77, 255, 0.1)',
          tension: 0.4,
          fill: true,
        },
        {
          label: 'Transaction Count',
          data: marketData?.transactions || [28, 48, 40, 19, 86, 27],
          borderColor: '#00ff88',
          backgroundColor: 'rgba(0, 255, 136, 0.1)',
          tension: 0.4,
          fill: true,
        },
      ],
    };
  }, [marketData]);
  
  // Risk radar chart
  const riskChartData = useMemo(() => {
    return {
      labels: ['Credit Risk', 'Market Risk', 'Operational Risk', 'Liquidity Risk', 'Quantum Risk'],
      datasets: [
        {
          label: 'Current Risk',
          data: [
            riskMetrics?.creditRisk || 0.3,
            riskMetrics?.marketRisk || 0.5,
            riskMetrics?.operationalRisk || 0.4,
            riskMetrics?.liquidityRisk || 0.2,
            riskMetrics?.quantumRisk || 0.1,
          ],
          backgroundColor: 'rgba(255, 23, 68, 0.2)',
          borderColor: '#ff1744',
          borderWidth: 2,
          pointBackgroundColor: '#ff1744',
        },
        {
          label: 'Threshold',
          data: [0.8, 0.8, 0.8, 0.8, 0.8],
          backgroundColor: 'rgba(0, 229, 255, 0.1)',
          borderColor: '#00e5ff',
          borderWidth: 1,
          pointBackgroundColor: '#00e5ff',
        },
      ],
    };
  }, [riskMetrics]);
  
  // Handle quantum pulse animation
  useEffect(() => {
    const interval = setInterval(() => {
      quantumPulse.set(Math.random());
    }, 1000);
    
    return () => clearInterval(interval);
  }, [quantumPulse]);
  
  // Subscribe to market data
  useEffect(() => {
    subscribeToMarket(['EUR/USD', 'GBP/USD', 'USD/JPY']);
  }, [subscribeToMarket]);
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="quantum-dashboard"
    >
      {/* Top Navigation Bar */}
      <QuantumNavbar
        user={user}
        notifications={notifications}
        alerts={alerts}
        systemStatus={systemStatus}
        quantumState={quantumState}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        onToggleDarkMode={() => setDarkMode(!darkMode)}
      />
      
      {/* Main Content */}
      <div className="dashboard-content">
        {/* Sidebar */}
        <QuantumSidebar
          open={sidebarOpen}
          activeView={activeView}
          onSelectView={setActiveView}
          user={user}
        />
        
        {/* Dashboard Grid */}
        <div className="dashboard-grid">
          {/* Row 1: Key Metrics */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="metrics-row"
          >
            <QuantumMetricCard
              title="Total Assets"
              value={`$${formatNumber(dashboardMetrics.totalAssets)}`}
              change="+2.3%"
              icon={<AccountBalanceWallet />}
              color="primary"
              trend="up"
            />
            
            <QuantumMetricCard
              title="Daily Volume"
              value={`$${formatNumber(dashboardMetrics.dailyVolume)}`}
              change="+15.7%"
              icon={<SwapHoriz />}
              color="secondary"
              trend="up"
            />
            
            <QuantumMetricCard
              title="Settlement Speed"
              value={`${dashboardMetrics.settlementSpeed}s`}
              change="-12%"
              icon={<Speed />}
              color="success"
              trend="down"
            />
            
            <QuantumMetricCard
              title="Security Score"
              value={`${dashboardMetrics.securityScore}/100`}
              change="+5"
              icon={<Lock />}
              color="quantum"
              trend="up"
            />
          </motion.div>
          
          {/* Row 2: Charts & Visualization */}
          <div className="charts-row">
            {/* Quantum State Visualization */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="chart-container quantum-chart"
            >
              <div className="chart-header">
                <QuantumSimulation className="chart-icon" />
                <h3>Quantum State Visualization</h3>
                <div className="chart-controls">
                  <button className="quantum-btn">Entangle</button>
                  <button className="quantum-btn">Measure</button>
                </div>
              </div>
              <div className="chart-body">
                <Line 
                  ref={quantumChartRef}
                  data={quantumChartData}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        display: false,
                      },
                      title: {
                        display: false,
                      },
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 1,
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)',
                        },
                      },
                      x: {
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)',
                        },
                      },
                    },
                  }}
                />
                <QuantumWaveVisualization />
              </div>
            </motion.div>
            
            {/* Market Data */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="chart-container market-chart"
            >
              <div className="chart-header">
                <ShowChart className="chart-icon" />
                <h3>Market Performance</h3>
                <div className="chart-controls">
                  <select className="time-selector">
                    <option>1D</option>
                    <option>1W</option>
                    <option>1M</option>
                    <option>1Y</option>
                  </select>
                </div>
              </div>
              <div className="chart-body">
                <Line 
                  ref={marketChartRef}
                  data={marketChartData}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        position: 'top',
                      },
                    },
                    scales: {
                      y: {
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)',
                        },
                      },
                      x: {
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)',
                        },
                      },
                    },
                  }}
                />
              </div>
            </motion.div>
          </div>
          
          {/* Row 3: Risk & Compliance */}
          <div className="risk-compliance-row">
            {/* Risk Radar */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="chart-container risk-chart"
            >
              <div className="chart-header">
                <Analytics className="chart-icon" />
                <h3>Risk Assessment</h3>
                <div className="risk-score">
                  <span className="score-label">Overall:</span>
                  <span className={`score-value ${getRiskClass(dashboardMetrics.riskScore)}`}>
                    {(dashboardMetrics.riskScore * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="chart-body">
                <Radar 
                  ref={riskChartRef}
                  data={riskChartData}
                  options={{
                    responsive: true,
                    scales: {
                      r: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                          stepSize: 0.2,
                        },
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)',
                        },
                      },
                    },
                  }}
                />
              </div>
            </motion.div>
            
            {/* Compliance Status */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="compliance-widget"
            >
              <div className="widget-header">
                <Security className="widget-icon" />
                <h3>Compliance Status</h3>
              </div>
              <div className="widget-body">
                <ComplianceStatus complianceStatus={complianceStatus} />
                <ComplianceAlerts alerts={complianceAlerts} />
              </div>
            </motion.div>
            
            {/* Quantum Security */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="security-widget"
            >
              <div className="widget-header">
                <Lock className="widget-icon" />
                <h3>Quantum Security</h3>
                <motion.div
                  animate={controls}
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: securityGlow,
                    marginLeft: 8,
                  }}
                />
              </div>
              <div className="widget-body">
                <QuantumSecurityStatus status={securityStatus} />
                <QuantumProofs proofs={quantumProofs} />
              </div>
            </motion.div>
          </div>
          
          {/* Row 4: Recent Transactions & Alerts */}
          <div className="transactions-alerts-row">
            {/* Recent Transactions */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="transactions-widget"
            >
              <div className="widget-header">
                <Payment className="widget-icon" />
                <h3>Recent Transactions</h3>
                <button className="view-all-btn">View All</button>
              </div>
              <div className="widget-body">
                <TransactionList transactions={transactions?.slice(0, 5) || []} />
              </div>
            </motion.div>
            
            {/* Active Alerts */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="alerts-widget"
            >
              <div className="widget-header">
                <Notifications className="widget-icon" />
                <h3>Active Alerts</h3>
                <span className="alert-count">{alerts.length}</span>
              </div>
              <div className="widget-body">
                <AlertList alerts={alerts} />
              </div>
            </motion.div>
            
            {/* Arbitrage Opportunities */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.9 }}
              className="arbitrage-widget"
            >
              <div className="widget-header">
                <TrendingUp className="widget-icon" />
                <h3>Arbitrage Opportunities</h3>
              </div>
              <div className="widget-body">
                <ArbitrageList opportunities={arbitrageOpportunities} />
              </div>
            </motion.div>
          </div>
        </div>
      </div>
      
      {/* Quantum Background Effects */}
      <QuantumBackground />
      
      {/* Snackbar for notifications */}
      <Snackbar 
        open={notifications.length > 0}
        autoHideDuration={6000}
        onClose={() => setNotifications([])}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity="info">
          {notifications[0]?.message}
        </Alert>
      </Snackbar>
    </motion.div>
  );
};

// Quantum Navbar Component
const QuantumNavbar = ({ 
  user, 
  notifications, 
  alerts, 
  systemStatus, 
  quantumState,
  onToggleSidebar,
  onToggleDarkMode 
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  
  return (
    <motion.nav 
      className="quantum-navbar"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100 }}
    >
      <div className="navbar-left">
        <button className="sidebar-toggle" onClick={onToggleSidebar}>
          <DashboardIcon />
        </button>
        
        <div className="navbar-brand">
          <div className="quantum-logo">
            <QuantumWaveVisualization small />
            <h1>Quantum Banking</h1>
          </div>
          <div className="system-status">
            <span className={`status-indicator ${systemStatus}`}>
              {systemStatus.toUpperCase()}
            </span>
            <span className={`quantum-status ${quantumState}`}>
              QUANTUM: {quantumState.toUpperCase()}
            </span>
          </div>
        </div>
      </div>
      
      <div className="navbar-center">
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search transactions, accounts, markets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button className="search-btn">
            <SearchIcon />
          </button>
        </div>
      </div>
      
      <div className="navbar-right">
        <button className="nav-btn">
          <Notifications />
          {alerts.length > 0 && (
            <span className="badge">{alerts.length}</span>
          )}
        </button>
        
        <button className="nav-btn">
          <Settings />
        </button>
        
        <button className="nav-btn" onClick={onToggleDarkMode}>
          <DarkModeToggle />
        </button>
        
        <div className="user-profile">
          <AccountCircle />
          <div className="user-info">
            <span className="user-name">{user?.name}</span>
            <span className="user-role">{user?.role}</span>
          </div>
        </div>
      </div>
    </motion.nav>
  );
};

// Quantum Sidebar Component
const QuantumSidebar = ({ open, activeView, onSelectView, user }) => {
  const menuItems = [
    { id: 'overview', label: 'Dashboard', icon: <DashboardIcon /> },
    { id: 'accounts', label: 'Accounts', icon: <AccountBalance /> },
    { id: 'payments', label: 'Payments', icon: <Payment /> },
    { id: 'trading', label: 'Trading', icon: <SwapHoriz /> },
    { id: 'analytics', label: 'Analytics', icon: <ShowChart /> },
    { id: 'risk', label: 'Risk Management', icon: <Analytics /> },
    { id: 'compliance', label: 'Compliance', icon: <Security /> },
    { id: 'quantum', label: 'Quantum Lab', icon: <QuantumSimulation /> },
    { id: 'reports', label: 'Reports', icon: <BarChart /> },
    { id: 'settings', label: 'Settings', icon: <Settings /> },
  ];
  
  return (
    <motion.aside 
      className="quantum-sidebar"
      initial={{ x: -300 }}
      animate={{ x: open ? 0 : -300 }}
      transition={{ type: 'spring', stiffness: 100 }}
    >
      <div className="sidebar-header">
        <div className="user-profile-summary">
          <AccountCircle className="profile-avatar" />
          <div className="profile-info">
            <h4>{user?.name}</h4>
            <p className="profile-email">{user?.email}</p>
            <p className="profile-role">{user?.role}</p>
          </div>
          <div className="profile-stats">
            <div className="stat">
              <span className="stat-label">Risk Level</span>
              <span className={`stat-value ${user?.riskLevel}`}>
                {user?.riskScore}/100
              </span>
            </div>
          </div>
        </div>
      </div>
      
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <motion.button
            key={item.id}
            className={`nav-item ${activeView === item.id ? 'active' : ''}`}
            onClick={() => onSelectView(item.id)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
            {item.id === 'quantum' && (
              <span className="quantum-badge">NEW</span>
            )}
          </motion.button>
        ))}
      </nav>
      
      <div className="sidebar-footer">
        <div className="quantum-status-indicator">
          <div className="quantum-pulse"></div>
          <span>Quantum Network Active</span>
        </div>
        <div className="system-info">
          <span>v2.0.0</span>
          <span>99.999% Uptime</span>
        </div>
      </div>
    </motion.aside>
  );
};

// Quantum Metric Card Component
const QuantumMetricCard = ({ title, value, change, icon, color, trend }) => {
  return (
    <motion.div 
      className={`metric-card ${color}`}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
    >
      <div className="metric-header">
        <div className="metric-icon">
          {icon}
        </div>
        <h4>{title}</h4>
      </div>
      
      <div className="metric-value">
        {value}
      </div>
      
      <div className="metric-footer">
        <span className={`metric-change ${trend}`}>
          {trend === 'up' ? '↗' : '↘'} {change}
        </span>
        <div className="metric-trend">
          <TrendIndicator trend={trend} />
        </div>
      </div>
      
      <div className="metric-wave"></div>
    </motion.div>
  );
};

// Quantum Wave Visualization Component
const QuantumWaveVisualization = ({ small = false }) => {
  const canvasRef = useRef();
  const [waveData, setWaveData] = useState([]);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Generate wave data
    const generateWave = () => {
      const points = [];
      for (let i = 0; i < width; i += 10) {
        const x = i;
        const y = height / 2 + Math.sin(i * 0.05 + Date.now() * 0.001) * 20;
        points.push({ x, y });
      }
      setWaveData(points);
    };
    
    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Draw wave
      ctx.beginPath();
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2;
      
      waveData.forEach((point, i) => {
        if (i === 0) {
          ctx.moveTo(point.x, point.y);
        } else {
          ctx.lineTo(point.x, point.y);
        }
      });
      
      ctx.stroke();
      
      // Draw quantum particles
      waveData.forEach((point, i) => {
        if (i % 5 === 0) {
          ctx.beginPath();
          ctx.fillStyle = '#7c4dff';
          ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    };
    
    const animate = () => {
      generateWave();
      draw();
      requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      // Cleanup
    };
  }, []);
  
  return (
    <canvas
      ref={canvasRef}
      width={small ? 100 : 400}
      height={small ? 40 : 100}
      className="quantum-wave-canvas"
    />
  );
};

// Compliance Status Component
const ComplianceStatus = ({ complianceStatus }) => {
  const requirements = [
    { id: 'kyc', label: 'KYC', status: complianceStatus?.kyc || 'pending' },
    { id: 'aml', label: 'AML', status: complianceStatus?.aml || 'pending' },
    { id: 'sanctions', label: 'Sanctions', status: complianceStatus?.sanctions || 'pending' },
    { id: 'tax', label: 'Tax', status: complianceStatus?.tax || 'pending' },
    { id: 'reporting', label: 'Reporting', status: complianceStatus?.reporting || 'pending' },
  ];
  
  return (
    <div className="compliance-status">
      {requirements.map((req) => (
        <div key={req.id} className="compliance-item">
          <span className="compliance-label">{req.label}</span>
          <span className={`compliance-badge ${req.status}`}>
            {req.status.toUpperCase()}
          </span>
        </div>
      ))}
    </div>
  );
};

// Quantum Security Status Component
const QuantumSecurityStatus = ({ status }) => {
  return (
    <div className="quantum-security-status">
      <div className="security-metric">
        <span className="metric-label">Quantum Encryption:</span>
        <span className="metric-value active">ACTIVE</span>
      </div>
      <div className="security-metric">
        <span className="metric-label">Key Rotation:</span>
        <span className="metric-value">
          {status?.lastKeyRotation || 'Never'}
        </span>
      </div>
      <div className="security-metric">
        <span className="metric-label">Threats Blocked:</span>
        <span className="metric-value">
          {status?.threatsBlocked || 0}
        </span>
      </div>
      <div className="security-metric">
        <span className="metric-label">Security Score:</span>
        <span className="metric-value score">
          {status?.score || 0}/100
        </span>
      </div>
    </div>
  );
};

// Transaction List Component
const TransactionList = ({ transactions }) => {
  return (
    <div className="transaction-list">
      {transactions.map((tx) => (
        <motion.div
          key={tx.id}
          className="transaction-item"
          whileHover={{ backgroundColor: 'rgba(124, 77, 255, 0.1)' }}
        >
          <div className="transaction-icon">
            {tx.type === 'sent' ? '↗' : '↘'}
          </div>
          <div className="transaction-details">
            <span className="transaction-amount">
              ${formatNumber(tx.amount)} {tx.currency}
            </span>
            <span className="transaction-parties">
              {tx.from} → {tx.to}
            </span>
          </div>
          <div className="transaction-meta">
            <span className="transaction-time">
              {formatTime(tx.timestamp)}
            </span>
            <span className={`transaction-status ${tx.status}`}>
              {tx.status}
            </span>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

// Alert List Component
const AlertList = ({ alerts }) => {
  return (
    <div className="alert-list">
      {alerts.map((alert) => (
        <motion.div
          key={alert.id}
          className={`alert-item ${alert.severity}`}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="alert-icon">
            {alert.severity === 'high' ? '⚠️' : 'ℹ️'}
          </div>
          <div className="alert-content">
            <span className="alert-title">{alert.title}</span>
            <span className="alert-message">{alert.message}</span>
          </div>
          <span className="alert-time">
            {formatTime(alert.timestamp)}
          </span>
        </motion.div>
      ))}
    </div>
  );
};

// Arbitrage List Component
const ArbitrageList = ({ opportunities }) => {
  return (
    <div className="arbitrage-list">
      {opportunities?.slice(0, 3).map((opp, index) => (
        <div key={index} className="arbitrage-item">
          <div className="arbitrage-path">
            <span>{opp.path}</span>
          </div>
          <div className="arbitrage-profit">
            <span className="profit-value">+{opp.profit}%</span>
            <span className="profit-time">{opp.executionTime}</span>
          </div>
          <button className="execute-btn">Execute</button>
        </div>
      ))}
    </div>
  );
};

// Quantum Background Component
const QuantumBackground = () => {
  const canvasRef = useRef();
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const particles = [];
    const particleCount = 50;
    
    // Initialize particles
    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2 + 1,
        speedX: Math.random() * 0.5 - 0.25,
        speedY: Math.random() * 0.5 - 0.25,
        color: `rgba(${Math.random() * 100 + 155}, ${Math.random() * 100 + 155}, 255, ${Math.random() * 0.3 + 0.1})`
      });
    }
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Update and draw particles
      particles.forEach(particle => {
        particle.x += particle.speedX;
        particle.y += particle.speedY;
        
        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;
        
        // Draw particle
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = particle.color;
        ctx.fill();
        
        // Draw connections
        particles.forEach(otherParticle => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < 100) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(124, 77, 255, ${0.2 * (1 - distance / 100)})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.stroke();
          }
        });
      });
      
      requestAnimationFrame(animate);
    };
    
    animate();
    
    // Handle resize
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);
  
  return (
    <canvas
      ref={canvasRef}
      className="quantum-background"
    />
  );
};

// Helper functions
function formatNumber(num) {
  if (num >= 1000000000) {
    return (num / 1000000000).toFixed(2) + 'B';
  }
  if (num >= 1000000) {
    return (num / 1000000).toFixed(2) + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(2) + 'K';
  }
  return num.toFixed(2);
}

function formatTime(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now - date;
  
  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
  return date.toLocaleDateString();
}

function calculateAverageSettlementSpeed(transactions) {
  if (!transactions || transactions.length === 0) return '0.0';
  
  const settled = transactions.filter(tx => tx.status === 'settled');
  if (settled.length === 0) return '0.0';
  
  const totalTime = settled.reduce((sum, tx) => sum + (tx.settlementTime || 0), 0);
  return (totalTime / settled.length / 1000).toFixed(1);
}

function calculateSecurityScore(alerts, securityStatus) {
  let score = 100;
  
  // Deduct for each high severity alert
  alerts.forEach(alert => {
    if (alert.severity === 'high') score -= 10;
    if (alert.severity === 'medium') score -= 5;
    if (alert.severity === 'low') score -= 2;
  });
  
  // Add for security features
  if (securityStatus?.quantumEncryption) score += 20;
  if (securityStatus?.multiSig) score += 15;
  if (securityStatus?.biometricAuth) score += 10;
  
  return Math.max(0, Math.min(100, score));
}

function calculateQuantumEfficiency(transactions) {
  if (!transactions || transactions.length === 0) return 0;
  
  const quantumTx = transactions.filter(tx => tx.quantumSecure);
  return (quantumTx.length / transactions.length * 100).toFixed(1);
}

function calculateSystemUptime() {
  // In production, this would come from monitoring system
  return '99.999';
}

function calculateSuccessRate(transactions) {
  if (!transactions || transactions.length === 0) return '100.0';
  
  const successful = transactions.filter(tx => tx.status === 'settled');
  return ((successful.length / transactions.length) * 100).toFixed(1);
}

function getRiskClass(score) {
  if (score < 0.3) return 'low';
  if (score < 0.7) return 'medium';
  return 'high';
}

// Context Providers
const App = () => {
  return (
    <ThemeProvider theme={quantumTheme}>
      <CssBaseline />
      <GlobalStyles styles={globalStyles} />
      <Router>
        <WebSocketProvider>
          <QuantumBankingProvider>
            <QuantumSecurityProvider>
              <MarketDataProvider>
                <RiskManagementProvider>
                  <ComplianceProvider>
                    <Routes>
                      <Route path="/" element={<QuantumEnterpriseDashboard />} />
                      <Route path="/dashboard" element={<QuantumEnterpriseDashboard />} />
                      {/* Add other routes here */}
                    </Routes>
                  </ComplianceProvider>
                </RiskManagementProvider>
              </MarketDataProvider>
            </QuantumSecurityProvider>
          </QuantumBankingProvider>
        </WebSocketProvider>
      </Router>
    </ThemeProvider>
  );
};

export default App;
