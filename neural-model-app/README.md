# 🧠 Neural Network Predictor App

Interactive React application for testing your trained neural network model with real-time visualizations.

## 🚀 Features

- **Interactive Input Interface**: 10 parameter input fields with validation
- **Real-time Predictions**: Mock and actual API integration support
- **Visual Analytics**: Bar charts showing class probability distributions
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Built with styled-components and beautiful gradients
- **API Integration**: Ready-to-use code for connecting to Flow Nexus

## 📋 Model Information

- **Model ID**: `model_1757095579212_5ee9v9o1t`
- **Architecture**: 256→128→64→32→10 Feedforward Neural Network
- **Training Accuracy**: 90.63%
- **Training Loss**: 0.268
- **Confidence**: 92.66%

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Quick Start

1. **Navigate to the app directory**:
   ```bash
   cd neural-model-app
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Open your browser** to `http://localhost:3000`

## 🔗 API Integration

### Current Status
The app currently uses **mock predictions** for demonstration. To connect to your actual neural network:

### Option 1: MCP Integration
```javascript
const mcpPredict = async () => {
  const result = await mcp.call("mcp__flow-nexus__neural_predict", {
    model_id: "model_1757095579212_5ee9v9o1t",
    input: inputs
  });
  setPrediction(result);
};
```

### Option 2: REST API
```javascript
const realPredict = async () => {
  const response = await fetch('/api/neural/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_id: "model_1757095579212_5ee9v9o1t",
      input: inputs
    })
  });
  const result = await response.json();
  setPrediction(result);
};
```

## 🎮 How to Use

1. **Enter Input Values**: Fill in the 10 numerical inputs (0.0 - 1.0)
2. **Generate Random**: Click "🎲 Random Values" for test data
3. **Run Prediction**: Click "🚀 Mock Predict" to see results
4. **View Results**: Analyze the probability distribution chart
5. **Clear & Repeat**: Use "🗑️ Clear All" to reset

## 📊 Understanding Results

The app displays:
- **Predicted Class**: The class with highest probability
- **Confidence**: Overall model confidence (0-100%)
- **Probability Chart**: Visual representation of all 10 class probabilities
- **Prediction Details**: Timestamp, input values, and prediction ID

## 🏗️ Architecture

```
neural-model-app/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── App.js             # Main React component
│   ├── App.css            # Styling
│   └── index.js           # React entry point
├── package.json           # Dependencies
└── README.md             # This file
```

## 📦 Dependencies

- **React 18**: Modern React with hooks
- **Styled Components**: CSS-in-JS styling
- **Recharts**: Beautiful, responsive charts
- **Axios**: HTTP client (ready for API calls)

## 🎨 Customization

### Colors & Theme
Modify the gradient and color scheme in `App.js`:
```javascript
const COLORS = ['#667eea', '#764ba2', '#28a745', ...];
```

### Model Information
Update model details at the top of `App.js`:
```javascript
const modelId = "your_model_id_here";
```

## 🚀 Production Deployment

### Build for Production
```bash
npm run build
```

### Deploy Options
- **Netlify**: Drag & drop the `build` folder
- **Vercel**: Connect your GitHub repo
- **GitHub Pages**: Use `gh-pages` package
- **Docker**: Create container with build files

## 📱 Mobile Support

The app is fully responsive and includes:
- Mobile-optimized input grid (2 columns on small screens)
- Touch-friendly buttons and interactions
- Responsive charts that adapt to screen size
- Optimized typography for mobile reading

## 🧪 Testing

### Run Tests
```bash
npm test
```

### Manual Testing Scenarios
1. **Valid Inputs**: Test with values 0.0-1.0
2. **Invalid Inputs**: Try negative numbers or >1.0
3. **Random Generation**: Verify random values are in range
4. **Mobile Response**: Test on different screen sizes
5. **Chart Interactions**: Hover over bars for tooltips

## 🔧 Troubleshooting

### Common Issues

1. **Dependencies not installing**: Clear npm cache with `npm cache clean --force`
2. **Port 3000 in use**: Change port with `PORT=3001 npm start`
3. **Chart not rendering**: Ensure recharts is properly installed
4. **Styling issues**: Clear browser cache and refresh

## 📈 Next Steps

### Enhancements You Can Add
- **Batch Prediction**: Support multiple input sets
- **History**: Save and display previous predictions
- **Export**: Download results as CSV/JSON
- **Comparison**: Compare multiple models
- **Real-time Updates**: WebSocket connection for live data

### API Enhancement
Replace mock prediction in `mockPredict()` function with actual Flow Nexus API calls.

## 📞 Support

For issues with:
- **App functionality**: Check browser console for errors
- **Neural model**: Use Flow Nexus MCP tools for debugging
- **API integration**: Verify model ID and endpoint URLs

---

**Your neural network model is ready for production use!** 🎯

Built with ❤️ using React and Flow Nexus AI Platform