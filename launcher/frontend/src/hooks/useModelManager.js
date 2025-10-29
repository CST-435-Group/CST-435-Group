import { useEffect, useRef } from 'react'

/**
 * Custom hook to manage model loading/unloading
 * Preloads the model when component mounts and unloads previous models
 *
 * @param {Object} currentAPI - The API object for the current project (annAPI, cnnAPI, or nlpAPI)
 * @param {Array} otherAPIs - Array of other project APIs to unload
 */
export function useModelManager(currentAPI, otherAPIs = []) {
  const hasPreloaded = useRef(false)

  useEffect(() => {
    let isMounted = true

    const loadModel = async () => {
      // Prevent double-loading in development strict mode
      if (hasPreloaded.current) return
      hasPreloaded.current = true

      try {
        // First, unload all other models to free memory
        console.log('🗑️ Unloading other models...')
        await Promise.allSettled(
          otherAPIs.map(api => api.unload().catch(err => console.log('Unload failed (expected):', err.message)))
        )

        // Then preload the current model
        if (isMounted) {
          console.log('🔄 Preloading current model...')
          await currentAPI.preload()
          console.log('✅ Model preloaded successfully')
        }
      } catch (error) {
        console.error('Error managing models:', error)
        // Don't throw - let the page load even if preload fails
      }
    }

    loadModel()

    // Cleanup function - don't unload on unmount to allow fast navigation
    return () => {
      isMounted = false
    }
  }, []) // Empty dependency array - only run on mount

  return null
}
