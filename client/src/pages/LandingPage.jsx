import React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  SparklesIcon,
  BarChart3Icon,
  MessageSquareIcon,
  FileTextIcon,
} from 'lucide-react'
export function LandingPage() {
  const navigate = useNavigate()
  const features = [
    {
      icon: <BarChart3Icon className="w-8 h-8" />,
      title: 'Topic Extraction',
      description:
        'Advanced NLP algorithms identify key themes and topics within your text',
    },
    {
      icon: <MessageSquareIcon className="w-8 h-8" />,
      title: 'Sentiment Visualization',
      description:
        'Visualize emotional tone and sentiment patterns across your content',
    },
    {
      icon: <FileTextIcon className="w-8 h-8" />,
      title: 'Comprehensive Reports',
      description:
        'Generate detailed analytical reports with actionable insights',
    },
  ]
  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <motion.div
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          transition={{
            duration: 0.8,
          }}
          className="text-center"
        >
          <div className="flex items-center justify-center mb-6">
            <SparklesIcon className="w-12 h-12 text-purple-400 mr-3" />
            <h1 className="text-6xl font-bold text-white">Narrative Nexus</h1>
          </div>
          <p className="text-2xl text-purple-200 mb-12">
            Unveiling stories hidden in your text
          </p>
          <motion.button
            whileHover={{
              scale: 1.05,
            }}
            whileTap={{
              scale: 0.95,
            }}
            onClick={() => navigate('/dashboard')}
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white px-8 py-4 rounded-full text-lg font-semibold shadow-lg hover:shadow-purple-500/50 transition-shadow"
          >
            Get Started
          </motion.button>
        </motion.div>
        {/* About Section */}
        <motion.div
          initial={{
            opacity: 0,
          }}
          animate={{
            opacity: 1,
          }}
          transition={{
            delay: 0.3,
            duration: 0.8,
          }}
          className="mt-32 max-w-4xl mx-auto"
        >
          <h2 className="text-4xl font-bold text-white text-center mb-8">
            About Us
          </h2>
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <p className="text-lg text-purple-100 leading-relaxed">
              Narrative Nexus is an advanced AI-powered text analytics platform
              that leverages cutting-edge Natural Language Processing (NLP) and
              Deep Learning technologies. Our system performs sophisticated
              topic modeling, sentiment analysis, and intelligent summarization
              to help you understand the narratives within your data. Transform
              raw text into actionable insights with our state-of-the-art
              analysis tools.
            </p>
          </div>
        </motion.div>
        {/* Features Section */}
        <motion.div
          initial={{
            opacity: 0,
          }}
          animate={{
            opacity: 1,
          }}
          transition={{
            delay: 0.6,
            duration: 0.8,
          }}
          className="mt-32"
        >
          <h2 className="text-4xl font-bold text-white text-center mb-16">
            Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{
                  opacity: 0,
                  y: 20,
                }}
                animate={{
                  opacity: 1,
                  y: 0,
                }}
                transition={{
                  delay: 0.8 + index * 0.2,
                  duration: 0.5,
                }}
                whileHover={{
                  y: -10,
                }}
                className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 text-center"
              >
                <div className="flex justify-center text-purple-400 mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-purple-200">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
