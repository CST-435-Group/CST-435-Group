# ✏️ Modification Tasks for Students

Complete **2-3** of these modifications before deploying your app.

---

## 🎯 Required: Choose 2-3 Modifications

### ⭐ Easy Tasks (Pick 1-2)

#### 1. Change Color Scheme
**File:** `frontend/tailwind.config.js`
**Time:** 5 minutes

Change the primary colors from blue to your favorite color.

**Instructions:**
1. Open `tailwind.config.js`
2. Find the `colors.primary` section
3. Change hex values to your preferred colors
4. Update Header.jsx to use new color for icon

**Example colors:**
- Purple: #8b5cf6
- Green: #10b981
- Pink: #ec4899
- Orange: #f97316

---

#### 2. Customize Header Text
**File:** `frontend/src/components/Header.jsx`
**Time:** 3 minutes

Change the app title and description to something unique.

**Instructions:**
1. Find the `<h1>` tag with "Sentiment Analyzer"
2. Change to your custom title
3. Update the description paragraph
4. Optionally add your name

**Ideas:**
- "Movie Review AI by [Your Name]"
- "Smart Sentiment Analyzer"
- "FilmFeel - AI Review Analysis"

---

#### 3. Add New Example Reviews
**File:** `backend/main.py`
**Time:** 10 minutes

Add 3-5 new example reviews in different sentiment categories.

**Instructions:**
1. Open `backend/main.py`
2. Find the `get_examples()` function
3. Add new reviews to each sentiment level
4. Make them creative and realistic!

---

### ⭐⭐ Medium Tasks (Pick 1)

#### 4. Add Character Limit
**Files:** `frontend/src/components/InputSection.jsx`
**Time:** 15 minutes

Add a 1000-character limit with visual feedback.

**What to implement:**
- Check text length in onChange handler
- Show error if exceeds limit
- Update counter to show "X/1000"
- Add warning color when near limit (e.g., 900+)

**Bonus:** Make the counter turn yellow at 800, orange at 900, red at limit.

---

#### 5. Add Clear Button
**File:** `frontend/src/components/InputSection.jsx`
**Time:** 10 minutes

Add a button to clear both input and results.

**What to implement:**
- New button next to "Analyze Sentiment"
- Click clears text, result, and error
- Styled with gray background
- Positioned properly in flex container

---

#### 6. Add More Metrics
**File:** `frontend/src/components/ResultsSection.jsx`
**Time:** 20 minutes

Add a new metric card showing additional information.

**Ideas:**
- Overall category (Positive/Negative/Neutral)
- Sentiment strength (based on confidence)
- Word count of analyzed text
- Processing time

---

#### 7. Improve Loading State
**Files:** `frontend/src/components/InputSection.jsx` and `ResultsSection.jsx`
**Time:** 15 minutes

Add a skeleton loader or progress indicator.

**What to implement:**
- Skeleton/shimmer effect while loading
- Progress bar
- Animated loading message
- Disable input during analysis

---

### ⭐⭐⭐ Advanced Tasks (Optional Challenge)

#### 8. Add Analysis History
**Files:** `frontend/src/App.jsx`, create new component
**Time:** 45 minutes

Store and display the last 5 analyses.

**What to implement:**
- State to store history array
- Add to history after each analysis
- Display in a new section
- Click to re-analyze
- Clear history button

**Bonus:** Use localStorage to persist between sessions.

---

#### 9. Add Dark Mode Toggle
**Files:** Multiple files, add new theme context
**Time:** 60 minutes

Implement a complete dark mode.

**What to implement:**
- Theme toggle button in header
- Dark styles for all components
- Use Tailwind's dark: prefix
- Save preference in localStorage
- Smooth transition animation

---

#### 10. Batch Analysis
**Files:** Create new component, update API service
**Time:** 90 minutes

Allow analyzing multiple reviews at once.

**What to implement:**
- Text area for multiple reviews (one per line)
- Call `/analyze/batch` endpoint
- Display results in table format
- Show average sentiment
- Export results as CSV

---

## 📝 Submission Checklist

After completing your modifications:

- [ ] All modifications are working locally
- [ ] Code is clean and commented
- [ ] Tested all functionality thoroughly
- [ ] Committed changes to Git
- [ ] Pushed to GitHub
- [ ] Updated README with your changes
- [ ] Deployed to Vercel
- [ ] Tested deployed version

---

## 💡 Tips

1. **Start Small:** Begin with easy tasks, then try harder ones
2. **Test Often:** Test after each change, don't wait until the end
3. **Use DevTools:** Press F12 to see console errors
4. **Read the Code:** Look at similar components for examples
5. **Be Creative:** Don't just copy - make it yours!
6. **Comment Your Code:** Explain what you changed and why

---

## 🎨 Design Ideas

### Color Schemes
- **Ocean:** Blues and teals (#0891b2, #06b6d4)
- **Forest:** Greens (#059669, #10b981)
- **Sunset:** Oranges and pinks (#f97316, #ec4899)
- **Royal:** Purples (#8b5cf6, #a855f7)
- **Fire:** Reds (#dc2626, #ef4444)

### Title Ideas
- "CineScore - AI Movie Review Analyzer"
- "SentiMeter - Measure Movie Sentiment"
- "FilmFeel - Smart Review Analysis"
- "MovieMood - AI-Powered Sentiment"
- "[Your Name]'s Sentiment Analyzer"

---

## 🐛 Common Issues

**Changes not showing up?**
- Save the file
- Check terminal for errors
- Refresh browser (Ctrl+R or Cmd+R)
- Try hard refresh (Ctrl+Shift+R)

**Syntax error?**
- Check all brackets match: {} [] ()
- Check all quotes match: "" '' ``
- Look at the line number in error message
- Use VS Code's error highlighting

**Import error?**
- Make sure file path is correct
- Check spelling of component name
- Verify export/import syntax matches

---

Good luck with your modifications! 🚀
