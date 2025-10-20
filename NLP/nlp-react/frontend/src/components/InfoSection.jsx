import { Info, Zap } from 'lucide-react';

function InfoSection() {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
      <div className="flex items-center gap-2 mb-4">
        <Info className="w-5 h-5 text-teal-600" />
        <h2 className="text-xl font-bold text-gray-800">Quick Info</h2>
      </div>

      <div className="space-y-4 text-sm">
        <div className="bg-teal-50 rounded-lg p-4 border border-teal-200">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-teal-600" />
            <h3 className="font-semibold text-gray-800">Model Performance</h3>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">Accuracy:</span>
              <span className="font-bold text-teal-700">86.5%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Algorithm:</span>
              <span className="font-medium text-gray-700">Logistic Regression</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Features:</span>
              <span className="font-medium text-gray-700">15,000 TF-IDF</span>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
          <h3 className="font-semibold text-gray-800 mb-2">Categories</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-lg">😊</span>
              <span className="text-xs text-gray-600">Ratings 4-5</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-lg">😐</span>
              <span className="text-xs text-gray-600">Rating 3</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-lg">😞</span>
              <span className="text-xs text-gray-600">Ratings 1-2</span>
            </div>
          </div>
        </div>

        <div className="pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500 italic text-center">
            Click "About NLP" tab for detailed explanation
          </p>
        </div>
      </div>
    </div>
  );
}

export default InfoSection;