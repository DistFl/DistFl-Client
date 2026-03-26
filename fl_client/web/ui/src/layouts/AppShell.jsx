import { useState, createContext, useContext } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useStatus, useToast } from '../hooks';
import Sidebar from '../components/Sidebar';
import Toast from '../components/Toast';

const ToastContext = createContext(null);
export const useAppToast = () => useContext(ToastContext);

export default function AppShell({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  const status = useStatus(3000);
  const { toasts, addToast, removeToast } = useToast();

  return (
    <ToastContext.Provider value={addToast}>
      <div className="flex h-screen overflow-hidden">
        <Sidebar
          collapsed={collapsed}
          onToggle={() => setCollapsed(!collapsed)}
          status={status}
        />
        <main
          className={`flex-1 overflow-y-auto transition-all duration-200 ${
            collapsed ? 'ml-16' : 'ml-60'
          }`}
        >
          <div className="max-w-[1400px] mx-auto px-6 py-6">
            {children}
          </div>
        </main>
      </div>

      {/* Toast stack */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map(t => (
          <Toast key={t.id} message={t.message} type={t.type} onClose={() => removeToast(t.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
