import React from 'react'
import { motion } from 'framer-motion'
import { FileTextIcon, DownloadIcon, CopyIcon } from 'lucide-react'

export function ReportSummary({ topics, method }) {
  const handleCopy = (text) => {
    navigator.clipboard.writeText(text)
    alert('Summary copied to clipboard!')
  }

  const handleDownloadReport = () => {
    const reportContent = `
NARRATIVE NEXUS - ANALYSIS REPORT
Method: ${method}
Total Topics: ${topics.length}
Generated: ${new Date().toLocaleString()}
${topics
  .map(
    (topic) => `
TOPIC ${topic.topic_id}
-----------------
Keywords: ${topic.keywords.slice(0, 10).join(', ')}
Sentiment: Positive ${topic.sentiment.positive}% | Neutral ${topic.sentiment.neutral}% | Negative ${topic.sentiment.negative}%
Summary:
${topic.summary}
`
  )
  .join('\n')}
    `.trim()

    const blob = new Blob([reportContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `narrative-nexus-report-${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-12"
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-white flex items-center">
          <FileTextIcon className="w-6 h-6 mr-2" />
          Report Summary
        </h2>
        <button
          onClick={handleDownloadReport}
          className="flex items-center bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <DownloadIcon className="w-4 h-4 mr-2" />
          Download Report
        </button>
      </div>

      <div className="space-y-6">
        {topics.map((topic, index) => (
          <motion.div
            key={topic.topic_id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20"
          >
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">
                Topic {topic.topic_id}
              </h3>
              <button
                onClick={() => handleCopy(topic.summary)}
                className="text-purple-300 hover:text-purple-200 transition-colors"
              >
                <CopyIcon className="w-5 h-5" />
              </button>
            </div>

            <div className="mb-4">
              <div className="flex gap-2 mb-2">
                <span className="text-sm text-purple-200">Keywords:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {topic.keywords.slice(0, 8).map((keyword, i) => (
                  <span
                    key={i}
                    className="bg-purple-500/30 text-purple-100 px-3 py-1 rounded-full text-sm"
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            </div>

            <div className="mb-4">
              <div className="flex gap-4 items-center">
                <div className="flex-1">
                  <div className="h-6 bg-slate-700 rounded-full overflow-hidden flex">
                    <div
                      className="bg-green-500 h-full transition-all"
                      style={{ width: `${topic.sentiment.positive}%` }}
                    />
                    <div
                      className="bg-blue-500 h-full transition-all"
                      style={{ width: `${topic.sentiment.neutral}%` }}
                    />
                    <div
                      className="bg-red-500 h-full transition-all"
                      style={{ width: `${topic.sentiment.negative}%` }}
                    />
                  </div>
                </div>
                <div className="flex gap-4 text-sm">
                  <span className="text-green-400">
                    {topic.sentiment.positive}%
                  </span>
                  <span className="text-blue-400">
                    {topic.sentiment.neutral}%
                  </span>
                  <span className="text-red-400">
                    {topic.sentiment.negative}%
                  </span>
                </div>
              </div>
            </div>

            <div>
              <p className="text-purple-100 leading-relaxed">{topic.summary}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}
