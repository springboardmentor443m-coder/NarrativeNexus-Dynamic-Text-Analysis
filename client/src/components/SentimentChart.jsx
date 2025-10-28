// import React from 'react'
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   YAxis,
//   CartesianGrid,
//   Tooltip,
//   Legend,
//   ResponsiveContainer,
// } from 'recharts'
// import { motion } from 'framer-motion'

// export function SentimentChart({ topics }) {
//   const data = topics.map((topic) => ({
//     name: `Topic ${topic.topic_id}`,
//     Positive: topic.sentiment.positive,
//     Neutral: topic.sentiment.neutral,
//     Negative: topic.sentiment.negative,
//   }))

//   return (
//     <motion.div
//       initial={{ opacity: 0, y: 20 }}
//       animate={{ opacity: 1, y: 0 }}
//       className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20"
//     >
//       <ResponsiveContainer width="100%" height={400}>
//         <BarChart data={data} layout="vertical">
//           <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
//           <XAxis type="number" stroke="#ffffff80" />
//           <YAxis dataKey="name" type="category" stroke="#ffffff80" />
//           <Tooltip
//             contentStyle={{
//               backgroundColor: '#1e293b',
//               border: '1px solid #ffffff20',
//               borderRadius: '8px',
//               color: '#fff',
//             }}
//           />
//           <Legend />
//           <Bar dataKey="Positive" fill="#10b981" />
//           <Bar dataKey="Neutral" fill="#6366f1" />
//           <Bar dataKey="Negative" fill="#ef4444" />
//         </BarChart>
//       </ResponsiveContainer>
//     </motion.div>
//   )
// }
import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";
import "./SentimentChart.css"; // 👈 Add this import

export function SentimentChart({ topics }) {
  const [activeTopic, setActiveTopic] = useState(null);

  const data = topics.map((topic) => ({
    name: `Topic ${topic.topic_id}`,
    Positive: topic.sentiment.positive,
    Neutral: topic.sentiment.neutral,
    Negative: topic.sentiment.negative,
  }));

  const handleBarClick = (data) => {
    setActiveTopic(data.name === activeTopic ? null : data.name);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl p-6 transition-all duration-300 ${
        activeTopic ? "border-2 border-indigo-400 shadow-[0_0_10px_rgba(99,102,241,0.5)] bg-transparent backdrop-blur-lg" : "border border-white/20 bg-transparent backdrop-blur-lg"
      }`}
    >
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={data}
          layout="vertical"
          onClick={handleBarClick}
          barCategoryGap="20%"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
          <XAxis type="number" stroke="#ffffff80" />
          <YAxis dataKey="name" type="category" stroke="#ffffff80" />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #ffffff20",
              borderRadius: "8px",
              color: "#fff",
            }}
          />
          <Legend />
          <Bar
            dataKey="Positive"
            fill="#10b981"
            activeBar={{ stroke: "#10b981", strokeWidth: 2, fill: "#10b981" }}
          />
          <Bar
            dataKey="Neutral"
            fill="#6366f1"
            activeBar={{ stroke: "#6366f1", strokeWidth: 2, fill: "#6366f1" }}
          />
          <Bar
            dataKey="Negative"
            fill="#ef4444"
            activeBar={{ stroke: "#ef4444", strokeWidth: 2, fill: "#ef4444" }}
          />
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
