import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';

// Components
import Layout from './components/Layout';

// Pages
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Transactions from './pages/Transactions';
import TransactionDetail from './pages/TransactionDetail';
import PredictSingle from './pages/PredictSingle';
import PredictBatch from './pages/PredictBatch';
import BehaviouralAnalysis from './pages/BehaviouralAnalysis';
import ModelComparison from './pages/ModelComparison';
import Alerts from './pages/Alerts';
import AuditLogs from './pages/AuditLogs';
import UserManagement from './pages/UserManagement';
import Profile from './pages/Profile';

const ProtectedRoute = ({ children, requiredRole }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-400 text-sm font-mono">
        Authenticating Secure Session...
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (requiredRole && user.role !== requiredRole && user.role !== 'admin') {
    return <Navigate to="/" replace />;
  }

  return children;
};

export const App = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Auth Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Protected Application Routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="transactions" element={<Transactions />} />
            <Route path="transactions/:id" element={<TransactionDetail />} />
            <Route path="transactions/customer/:ccNum" element={<BehaviouralAnalysis />} />
            <Route path="predict-single" element={<PredictSingle />} />
            <Route path="predict-batch" element={<PredictBatch />} />
            <Route path="behavioural" element={<BehaviouralAnalysis />} />
            <Route path="models" element={<ModelComparison />} />
            <Route path="alerts" element={<Alerts />} />
            <Route path="audit" element={<AuditLogs />} />
            <Route
              path="users"
              element={
                <ProtectedRoute requiredRole="admin">
                  <UserManagement />
                </ProtectedRoute>
              }
            />
            <Route path="profile" element={<Profile />} />
          </Route>

          {/* Catch-all redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
