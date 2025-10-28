import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeftIcon } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { FileUpload } from '../components/FileUpload'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { WordCloudChart } from '../components/WordCloudChart'
import { SentimentChart } from '../components/SentimentChart'
import { TopicDistributionChart } from '../components/TopicDistributionChart'
import { ReportSummary } from '../components/ReportSummary'

export function Dashboard() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analysisData, setAnalysisData] = useState(null)
  const [error, setError] = useState(null)

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please upload a file first')
      return
    }
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const data = await response.json()
      setAnalysisData(data)
    } catch (err) {
      setError('Failed to analyze file. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={() => navigate('/')}
            className="flex items-center text-purple-200 hover:text-white transition-colors"
          >
            <ArrowLeftIcon className="w-5 h-5 mr-2" />
            Back to Home
          </button>
          <h1 className="text-3xl font-bold text-white">Analysis Dashboard</h1>
          <div className="w-32"></div>
        </div>

        {/* Upload Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <FileUpload file={file} setFile={setFile} />

          {file && (
            <div className="text-center mt-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleAnalyze}
                disabled={loading}
                className="bg-gradient-to-r from-purple-500 to-blue-500 text-white px-8 py-3 rounded-full text-lg font-semibold shadow-lg hover:shadow-purple-500/50 transition-shadow disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </motion.button>
            </div>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-200 text-center">
              {error}
            </div>
          )}
        </motion.div>

        {/* Loading State */}
        {loading && <LoadingSpinner />}

        {/* Results Section */}
        {analysisData && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {/* Word Clouds */}
            <div className="mb-12">
              <h2 className="text-2xl font-bold text-white mb-6">
                Topic Word Clouds
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {analysisData.topics.map((topic) => (
                  <WordCloudChart key={topic.topic_id} topic={topic} />
                ))}
              </div>
            </div>

            {/* Sentiment Chart */}
            <div className="mb-12">
              <h2 className="text-2xl font-bold text-white mb-6">
                Sentiment Distribution
              </h2>
              <SentimentChart topics={analysisData.topics} />
            </div>

            {/* Topic Distribution */}
            <div className="mb-12">
              <h2 className="text-2xl font-bold text-white mb-6">
                Topic Prevalence
              </h2>
              <TopicDistributionChart topics={analysisData.topics} />
            </div>

            {/* Report Summary */}
            <ReportSummary
              topics={analysisData.topics}
              method={analysisData.method}
            />
          </motion.div>
        )}
      </div>
    </div>
  )
}
