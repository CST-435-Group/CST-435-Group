import { Brain, BookOpen, Target, TrendingUp, Sparkles, CheckCircle } from 'lucide-react';

function AboutNLP() {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-100">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-8 h-8 text-teal-600" />
        <h2 className="text-3xl font-bold text-gray-800">
          About Natural Language Processing (NLP)
        </h2>
      </div>

      {/* Introduction */}
      <div className="mb-8">
        <p className="text-lg text-gray-700 leading-relaxed">
          This project demonstrates <strong>Natural Language Processing (NLP)</strong>, a branch of 
          artificial intelligence that helps computers understand, interpret, and generate human language.
        </p>
      </div>

      {/* What This System Does */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          <Target className="w-6 h-6 text-teal-600" />
          <h3 className="text-2xl font-bold text-gray-800">What This System Does</h3>
        </div>
        <div className="bg-teal-50 rounded-xl p-6 border border-teal-200">
          <p className="text-gray-700 mb-4">
            Our sentiment analyzer automatically reads hospital reviews written by patients and 
            determines whether they express <strong className="text-green-600">positive</strong>, 
            <strong className="text-yellow-600"> neutral</strong>, or 
            <strong className="text-red-600"> negative</strong> sentiment.
          </p>
          <p className="text-gray-700">
            This helps healthcare administrators quickly understand patient satisfaction without 
            manually reading thousands of reviews.
          </p>
        </div>
      </div>

      {/* NLP Techniques Used */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-6 h-6 text-teal-600" />
          <h3 className="text-2xl font-bold text-gray-800">NLP Techniques We Use</h3>
        </div>
        <div className="space-y-4">
          <div className="flex gap-4 items-start">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-1" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">Tokenization</h4>
              <p className="text-gray-600">
                Breaking text into individual words. Example: "Great hospital!" → ["Great", "hospital"]
              </p>
            </div>
          </div>

            {/* Data Collection & Cleaning */}
            <div className="mb-8">
              <div className="flex items-center gap-2 mb-4">
                <BookOpen className="w-6 h-6 text-teal-600" />
                <h3 className="text-2xl font-bold text-gray-800">Data & Cleaning</h3>
              </div>
              <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 space-y-3">
                <p className="text-gray-700">
                  This project uses hospital review data (CSV) that must be cleaned before training or
                  inference. The cleaning steps are implemented in <code>data/clean_data.py</code> and in
                  backend preprocessors. Key steps include:
                </p>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li>Drop rows with missing review text or essential labels (Feedback, Ratings or Sentiment Label).</li>
                  <li>Normalize text to lowercase and remove HTML, URLs and email addresses.</li>
                  <li>Handle contractions and negations ("can't" → "cannot", "didn't" → "did not") so sentiment is preserved.</li>
                  <li>Remove punctuation and extra whitespace, then tokenize and lemmatize words (NLTK lemmatizer used in classical pipeline).</li>
                  <li>Carefully manage stop words: keep negation words (not, never) and sentiment-bearing tokens (good, bad).</li>
                  <li>Map labels to modeling scheme: some datasets use Ratings (1-5) or binary Sentiment Label (0/1). The code maps these to either a 3-class scheme (negative/neutral/positive) or a 7-point scale when relevant.</li>
                  <li>Save the cleaned CSV as <code>data/hospital_cleaned.csv</code> for reproducible training and evaluation.</li>
                </ul>
                <p className="text-sm text-gray-500">
                  See <code>data/clean_data.py</code> and the backend files <code>main_sentiment.py</code> and
                  <code>sentiment_model.py</code> for the exact cleaning rules and label mappings used by the app.
                </p>
              </div>
            </div>

            {/* Labeling and neutral-handling note */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <BookOpen className="w-5 h-5 text-teal-600" />
                <h4 className="text-lg font-semibold text-gray-800">Labels & how "Neutral" is handled</h4>
              </div>
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-200 text-gray-700">
                <p className="mb-2">
                  Many hospital datasets only include <strong>binary</strong> sentiment labels (for example <code>0</code> = negative, <code>1</code> = positive).
                  That means there is no explicit "neutral" label in the raw data, so the app applies simple, transparent rules to infer neutrality when needed.
                </p>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li><strong>Ratings mapping:</strong> When raw ratings (1–5) exist we map them to a numeric sentiment score and treat middle values as neutral.</li>
                  <li><strong>Model-predicted neutral:</strong> Our 3-class transformer may predict a neutral class directly; when it does we keep that as neutral.</li>
                  <li><strong>Heuristics:</strong> Short heuristics inspect the text for neutral indicators (words like "okay", "average", "nothing special") and lower-confidence predictions to prefer neutral.</li>
                  <li><strong>Confidence thresholds:</strong> Very low confidence or conflicting signals (mixed words) are treated conservatively as neutral rather than forcing positive/negative.</li>
                </ul>
                <p className="text-sm text-gray-500 mt-2">
                  These choices are pragmatic: they make the UI and statistics more stable when the source labels are coarse (0/1). If you prefer a stricter mapping (no inferred neutrals) or a different heuristic, I can update the rules or expose a switch in the UI.
                </p>
              </div>
            </div>

          <div className="flex gap-4 items-start">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-1" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">Lemmatization</h4>
              <p className="text-gray-600">
                Converting words to their root form. Example: "amazing, amazed" → "amaze"
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-1" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">Stop Word Removal</h4>
              <p className="text-gray-600">
                Removing common words like "the", "is", "at" that don't add sentiment meaning.
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-1" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">TF-IDF (Term Frequency-Inverse Document Frequency)</h4>
              <p className="text-gray-600">
                Measuring word importance. Frequent words in one review but rare overall get higher scores.
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-1" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">Machine Learning Classification</h4>
              <p className="text-gray-600">
                Training algorithms (Logistic Regression, SVM) on labeled data to predict sentiment.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Why This Matters */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-6 h-6 text-teal-600" />
          <h3 className="text-2xl font-bold text-gray-800">Why This Matters</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded-lg p-5 border border-blue-200">
            <h4 className="font-semibold text-blue-800 mb-2">🏥 Healthcare Quality</h4>
            <p className="text-gray-700 text-sm">
              Helps hospitals identify problems quickly and improve patient care based on feedback.
            </p>
          </div>

          <div className="bg-green-50 rounded-lg p-5 border border-green-200">
            <h4 className="font-semibold text-green-800 mb-2">⏱️ Time Savings</h4>
            <p className="text-gray-700 text-sm">
              Analyzes thousands of reviews in seconds instead of hours of manual reading.
            </p>
          </div>

          <div className="bg-purple-50 rounded-lg p-5 border border-purple-200">
            <h4 className="font-semibold text-purple-800 mb-2">📊 Data-Driven Decisions</h4>
            <p className="text-gray-700 text-sm">
              Provides quantifiable metrics (86% positive, 10% negative) for strategic planning.
            </p>
          </div>

          <div className="bg-orange-50 rounded-lg p-5 border border-orange-200">
            <h4 className="font-semibold text-orange-800 mb-2">🎯 Early Detection</h4>
            <p className="text-gray-700 text-sm">
              Identifies negative trends early so problems can be addressed before escalating.
            </p>
          </div>
        </div>
      </div>

      {/* Real World Applications */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="w-6 h-6 text-teal-600" />
          <h3 className="text-2xl font-bold text-gray-800">Real-World Applications</h3>
        </div>
        <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
          <ul className="space-y-3">
            <li className="flex items-start gap-3">
              <span className="text-teal-600 font-bold">•</span>
              <span className="text-gray-700"><strong>Social Media Monitoring:</strong> Companies track brand sentiment on Twitter, Facebook</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-teal-600 font-bold">•</span>
              <span className="text-gray-700"><strong>Customer Service:</strong> Automatically route angry customers to priority support</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-teal-600 font-bold">•</span>
              <span className="text-gray-700"><strong>Product Reviews:</strong> Amazon, Yelp use NLP to summarize millions of reviews</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-teal-600 font-bold">•</span>
              <span className="text-gray-700"><strong>Market Research:</strong> Analyze survey responses to understand consumer opinions</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-teal-600 font-bold">•</span>
              <span className="text-gray-700"><strong>Healthcare:</strong> Monitor patient satisfaction and safety concerns</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Model Performance */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-6 h-6 text-teal-600" />
          <h3 className="text-2xl font-bold text-gray-800">Our Model's Performance</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-5 border border-green-300 text-center">
            <div className="text-4xl font-bold text-green-700 mb-1">86.5%</div>
            <div className="text-sm text-green-800 font-medium">Overall Accuracy</div>
          </div>
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-5 border border-blue-300 text-center">
            <div className="text-4xl font-bold text-blue-700 mb-1">996</div>
            <div className="text-sm text-blue-800 font-medium">Training Reviews</div>
          </div>
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-5 border border-purple-300 text-center">
            <div className="text-4xl font-bold text-purple-700 mb-1">15,000</div>
            <div className="text-sm text-purple-800 font-medium">Features Analyzed</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AboutNLP;