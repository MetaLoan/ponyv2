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
  if (tasks.length === 0) return null;

  return (
    <div style={{ marginTop: "2rem", padding: "1rem", border: "1px solid #444", borderRadius: "8px", background: "#1e1e1e" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3>🚀 Task Center</h3>
        <button onClick={clearTasks} style={{ background: "transparent", border: "1px solid #666", padding: "4px 8px", cursor: "pointer", color: "#ccc" }}>
          Clear All
        </button>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "1rem" }}>
        {tasks.map((t) => (
          <div key={t.id} style={{ padding: "10px", border: "1px solid #333", borderRadius: "4px", background: "#252525" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.9em" }}>
              <span style={{ color: "#aaa" }}>{new Date(t.timestamp).toLocaleTimeString()} - {t.mode}</span>
              <strong style={{ 
                color: t.status === "COMPLETED" ? "#4caf50" : t.status === "FAILED" ? "#f44336" : "#2196f3" 
              }}>
                {t.status}
              </strong>
            </div>
            <p style={{ fontSize: "0.85em", color: "#ddd", margin: "5px 0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {t.prompt}
            </p>
            {t.status === "COMPLETED" && t.result && t.result.final_video_url && (
              <a href={t.result.final_video_url} target="_blank" rel="noreferrer" style={{ color: "#4caf50", fontSize: "0.9em" }}>
                ▶️ Watch Video
              </a>
            )}
            {t.status === "COMPLETED" && t.result && !t.result.final_video_url && t.result.final_url && (
              <a href={t.result.final_url} target="_blank" rel="noreferrer" style={{ color: "#4caf50", fontSize: "0.9em" }}>
                🖼️ View Image
              </a>
            )}
            {t.status === "FAILED" && t.result?.error && (
              <p style={{ color: "#f44336", fontSize: "0.8em", margin: "5px 0" }}>{t.result.error}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
