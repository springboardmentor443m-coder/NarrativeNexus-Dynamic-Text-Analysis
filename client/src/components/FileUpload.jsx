import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { UploadCloudIcon, FileIcon, XIcon } from 'lucide-react'
import { motion } from 'framer-motion'

export function FileUpload({ file, setFile }) {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0])
      }
    },
    [setFile]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
  })

  return (
    <div className="max-w-2xl mx-auto">
      {!file ? (
        <motion.div
          {...getRootProps()}
          whileHover={{ scale: 1.02 }}
          className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-purple-400 bg-purple-500/20'
              : 'border-purple-300/50 bg-white/5 hover:bg-white/10'
          }`}
        >
          <input {...getInputProps()} />
          <UploadCloudIcon className="w-16 h-16 text-purple-400 mx-auto mb-4" />
          <p className="text-xl text-white mb-2">
            {isDragActive ? 'Drop your file here' : 'Drag & drop your file here'}
          </p>
          <p className="text-purple-200">or click to browse</p>
          <p className="text-sm text-purple-300 mt-4">
            Supported formats: .txt, .pdf
          </p>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <FileIcon className="w-8 h-8 text-purple-400 mr-4" />
              <div>
                <p className="text-white font-semibold">{file.name}</p>
                <p className="text-purple-200 text-sm">
                  {(file.size / 1024).toFixed(2)} KB
                </p>
              </div>
            </div>
            <button
              onClick={() => setFile(null)}
              className="text-red-400 hover:text-red-300 transition-colors"
            >
              <XIcon className="w-6 h-6" />
            </button>
          </div>
        </motion.div>
      )}
    </div>
  )
}
