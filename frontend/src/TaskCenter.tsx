import React, { useState, useEffect } from "react";

export interface Task {
  id: string;
  job_id?: string;
  prompt_id?: string;
  mode: string;
  prompt: string;
  status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | string;
  result?: any;
  timestamp: number;
}

export function useTasks() {
  const [tasks, setTasks] = useState<Task[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem("ponyv2_tasks");
    if (saved) {
      try {
        setTasks(JSON.parse(saved));
      } catch (e) {}
    }
  }, []);

  const saveTasks = (newTasks: Task[]) => {
    setTasks(newTasks);
    localStorage.setItem("ponyv2_tasks", JSON.stringify(newTasks));
  };

  const addTask = (task: Task) => {
    saveTasks([task, ...tasks]);
  };

  const updateTask = (id: string, updates: Partial<Task>) => {
    saveTasks(tasks.map((t) => (t.id === id ? { ...t, ...updates } : t)));
  };

  const clearTasks = () => {
    saveTasks([]);
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      const pendingTasks = tasks.filter((t) => t.status === "IN_QUEUE" || t.status === "IN_PROGRESS");
      for (const t of pendingTasks) {
        if (!t.job_id) continue;
        try {
          const res = await fetch(`/api/status?job_id=${t.job_id}`);
          const data = await res.json();
          if (data.status !== t.status || data.ok) {
            updateTask(t.id, {
              status: data.status || "COMPLETED",
              result: data,
            });
          }
        } catch (e) {
          console.error("Poll error for task", t.id, e);
        }
      }
    }, 10000);
    return () => clearInterval(interval);
  }, [tasks]);

  return { tasks, addTask, clearTasks };
}

export function TaskCenter({ tasks, clearTasks }: { tasks: Task[]; clearTasks: () => void }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const pendingCount = tasks.filter((t) => t.status === "IN_QUEUE" || t.status === "IN_PROGRESS").length;

  if (!isExpanded) {
    return (
      <div 
        onClick={() => setIsExpanded(true)}
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          width: "60px",
          height: "60px",
          borderRadius: "30px",
          background: "#2196f3",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          cursor: "pointer",
          boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
          zIndex: 9999,
          transition: "transform 0.2s"
        }}
      >
        <span style={{ fontSize: "24px" }}>🚀</span>
        {pendingCount > 0 && (
          <div style={{
            position: "absolute",
            top: "-5px",
            right: "-5px",
            background: "#f44336",
            color: "white",
            borderRadius: "50%",
            width: "24px",
            height: "24px",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            fontSize: "12px",
            fontWeight: "bold",
            border: "2px solid #1e1e1e"
          }}>
            {pendingCount}
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ 
      position: "fixed",
      bottom: "20px",
      right: "20px",
      width: "350px",
      maxHeight: "80vh",
      display: "flex",
      flexDirection: "column",
      border: "1px solid #444", 
      borderRadius: "12px", 
      background: "#1e1e1e",
      boxShadow: "0 10px 30px rgba(0,0,0,0.8)",
      zIndex: 9999,
      overflow: "hidden"
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "15px", borderBottom: "1px solid #333", background: "#252525" }}>
        <h3 style={{ margin: 0, fontSize: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
          🚀 Task Center 
          {pendingCount > 0 && <span style={{ background: "#2196f3", padding: "2px 8px", borderRadius: "10px", fontSize: "12px" }}>{pendingCount} Running</span>}
        </h3>
        <div style={{ display: "flex", gap: "10px" }}>
          <button onClick={clearTasks} style={{ background: "transparent", border: "1px solid #666", padding: "4px 8px", borderRadius: "4px", cursor: "pointer", color: "#ccc", fontSize: "12px" }}>
            Clear All
          </button>
          <button onClick={() => setIsExpanded(false)} style={{ background: "transparent", border: "none", cursor: "pointer", color: "#ccc", fontSize: "16px", padding: "0 4px" }}>
            ✕
          </button>
        </div>
      </div>
      
      <div style={{ display: "flex", flexDirection: "column", gap: "10px", padding: "15px", overflowY: "auto", flex: 1 }}>
        {tasks.length === 0 ? (
          <div style={{ padding: "30px 10px", textAlign: "center" }}>
            <span style={{ fontSize: "40px", display: "block", marginBottom: "10px", opacity: 0.5 }}>📭</span>
            <p style={{ color: "#888", fontStyle: "italic", margin: 0, fontSize: "14px" }}>
              No tasks in queue.<br/>Generate something to see it here!
            </p>
          </div>
        ) : tasks.map((t) => (
          <div key={t.id} style={{ padding: "12px", border: "1px solid #333", borderRadius: "8px", background: "#2a2a2a" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85em", marginBottom: "6px" }}>
              <span style={{ color: "#aaa" }}>{new Date(t.timestamp).toLocaleTimeString()} - {t.mode}</span>
              <strong style={{ 
                color: t.status === "COMPLETED" ? "#4caf50" : t.status === "FAILED" ? "#f44336" : "#2196f3" 
              }}>
                {t.status === "IN_QUEUE" || t.status === "IN_PROGRESS" ? (
                  <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                    <span className="spinner" style={{ width: "10px", height: "10px", border: "2px solid #2196f3", borderTopColor: "transparent", borderRadius: "50%", display: "inline-block", animation: "spin 1s linear infinite" }}></span>
                    {t.status}
                  </span>
                ) : t.status}
              </strong>
            </div>
            <p style={{ fontSize: "0.85em", color: "#eee", margin: "0 0 8px 0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={t.prompt}>
              {t.prompt}
            </p>
            {t.status === "COMPLETED" && t.result && t.result.final_video_url && (
              <a href={t.result.final_video_url} target="_blank" rel="noreferrer" style={{ color: "#4caf50", fontSize: "0.9em", display: "inline-block", background: "rgba(76, 175, 80, 0.1)", padding: "4px 8px", borderRadius: "4px", textDecoration: "none" }}>
                ▶️ Watch Video
              </a>
            )}
            {t.status === "COMPLETED" && t.result && !t.result.final_video_url && t.result.final_url && (
              <a href={t.result.final_url} target="_blank" rel="noreferrer" style={{ color: "#4caf50", fontSize: "0.9em", display: "inline-block", background: "rgba(76, 175, 80, 0.1)", padding: "4px 8px", borderRadius: "4px", textDecoration: "none" }}>
                🖼️ View Image
              </a>
            )}
            {t.status === "FAILED" && t.result?.error && (
              <p style={{ color: "#f44336", fontSize: "0.8em", margin: "5px 0", background: "rgba(244, 67, 54, 0.1)", padding: "6px", borderRadius: "4px" }}>
                {t.result.error}
              </p>
            )}
          </div>
        ))}
      </div>
      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
