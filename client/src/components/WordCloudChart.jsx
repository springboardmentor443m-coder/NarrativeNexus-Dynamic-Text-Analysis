// import React, { useEffect, useRef } from "react";
// import * as d3 from "d3";
// import cloud from "d3-cloud";
// import { motion } from "framer-motion";

// export function WordCloudChart({ topic }) {
//   const svgRef = useRef(null);

//   useEffect(() => {
//     if (!svgRef.current || !topic.keywords.length) return;

//     const container = svgRef.current;
//     const width = container.clientWidth || 500;
//     const height = 300;

//     const svg = d3.select(container);
//     svg.selectAll("*").remove();

//     const words = topic.keywords.slice(0, 25).map((word, i) => ({
//       text: word,
//       size: 35 - i * 1.2,
//     }));

//     const color = d3
//       .scaleLinear()
//       .domain([0, words.length])
//       .range(["#ffffff", "#d0c8ff"]);

//     // Generate layout using d3-cloud
//     const layout = cloud()
//       .size([width, height])
//       .words(words)
//       .padding(5)
//       .rotate(() => 0) // keep all horizontal
//       .font("'Inter', sans-serif")
//       .fontSize((d) => d.size)
//       .on("end", draw);

//     layout.start();

//     function draw(data) {
//       const g = svg
//         .append("g")
//         .attr("transform", `translate(${width / 2},${height / 2})`);

//       g.selectAll("text")
//         .data(data)
//         .enter()
//         .append("text")
//         .style("font-family", "'Inter', sans-serif")
//         .style("font-weight", 600)
//         .style("fill", (d, i) => color(i))
//         .style("text-shadow", "0px 0px 6px rgba(255,255,255,0.5)")
//         .attr("text-anchor", "middle")
//         .attr("transform", (d) => `translate(${d.x},${d.y})rotate(${d.rotate})`)
//         .style("font-size", (d) => `${d.size}px`)
//         .text((d) => d.text);
//     }
//   }, [topic]);

//   return (
//     <motion.div
//       initial={{ opacity: 0, scale: 0.9 }}
//       animate={{ opacity: 1, scale: 1 }}
//       className="bg-gradient-to-br from-purple-800 to-indigo-800 rounded-2xl p-6 shadow-lg border border-white/10"
//     >
//       <h3 className="text-xl font-semibold text-white mb-4">
//         Topic {topic.topic_id}
//       </h3>
//       <svg ref={svgRef} width="100%" height="300" />
//     </motion.div>
//   );
// }

import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import cloud from "d3-cloud";
import { motion } from "framer-motion";

export function WordCloudChart({ topic }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !topic.keywords.length) return;

    const container = svgRef.current;
    const width = container.clientWidth || 500;
    const height = 300;

    const svg = d3.select(container);
    svg.selectAll("*").remove();

    const words = topic.keywords.slice(0, 25).map((word, i) => ({
      text: word,
      size: 35 - i * 1.2,
    }));

    const color = d3
      .scaleLinear()
      .domain([0, words.length])
      .range(["#ffffff", "#d0c8ff"]);

    // 💡 Increased padding from 5 → 10 for better spacing
    const layout = cloud()
      .size([width, height])
      .words(words)
      .padding(18) // spacing between words
      .rotate(() => 0)
      .font("'Inter', sans-serif")
      .fontSize((d) => d.size)
      .on("end", draw);

    layout.start();

    function draw(data) {
      const g = svg
        .append("g")
        .attr("transform", `translate(${width / 2},${height / 2})`);

      g.selectAll("text")
        .data(data)
        .enter()
        .append("text")
        .style("font-family", "'Inter', sans-serif")
        .style("font-weight", 600)
        .style("fill", (d, i) => color(i))
        .style("text-shadow", "0px 0px 6px rgba(255,255,255,0.5)")
        .attr("text-anchor", "middle")
        .attr("transform", (d) => `translate(${d.x},${d.y})rotate(${d.rotate})`)
        .style("font-size", (d) => `${d.size}px`)
        .text((d) => d.text);
    }
  }, [topic]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-gradient-to-br from-purple-800 to-indigo-800 rounded-2xl p-6 shadow-lg border border-white/10"
    >
      <h3 className="text-xl font-semibold text-white mb-4">
        Topic {topic.topic_id}
      </h3>
      <svg ref={svgRef} width="100%" height="300" />
    </motion.div>
  );
}
