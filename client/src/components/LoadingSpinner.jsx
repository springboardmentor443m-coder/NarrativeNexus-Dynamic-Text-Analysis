import React from 'react'
import { motion } from 'framer-motion'
export function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <motion.div
        animate={{
          rotate: 360,
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: 'linear',
        }}
        className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full"
      />
      <p className="text-white mt-4 text-lg">Analyzing your text...</p>
      <p className="text-purple-200 text-sm mt-2">
        This may take a few moments
      </p>
    </div>
  )
}
