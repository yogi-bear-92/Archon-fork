import React, { useState } from 'react';
import styled from 'styled-components';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './App.css';

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Arial', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
`;

const Header = styled.h1`
  text-align: center;
  color: white;
  margin-bottom: 10px;
  font-size: 2.5rem;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
`;

const Subtitle = styled.p`
  text-align: center;
  color: rgba(255,255,255,0.9);
  margin-bottom: 30px;
  font-size: 1.1rem;
`;

const Card = styled.div`
  background: white;
  border-radius: 15px;
  padding: 25px;
  margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  backdrop-filter: blur(10px);
`;

const InputGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 15px;
  margin-bottom: 20px;
  
  @media (max-width: 768px) {
    grid-template-columns: repeat(2, 1fr);
  }
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
`;

const Label = styled.label`
  font-weight: bold;
  color: #333;
  margin-bottom: 5px;
  font-size: 0.9rem;
`;

const Input = styled.input`
  padding: 10px;
  border: 2px solid #ddd;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: #667eea;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 10px;
  justify-content: center;
  margin-bottom: 20px;
  flex-wrap: wrap;
`;

const Button = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  }
`;

const PredictButton = styled(Button)`
  background: linear-gradient(45deg, #667eea, #764ba2);
  color: white;
  
  &:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
  }
`;

const RandomButton = styled(Button)`
  background: #28a745;
  color: white;
`;

const ClearButton = styled(Button)`
  background: #dc3545;
  color: white;
`;

const APIButton = styled(Button)`
  background: #17a2b8;
  color: white;
`;

const ResultCard = styled(Card)`
  ${props => props.show ? 'display: block;' : 'display: none;'}
`;

const StatusMessage = styled.div`
  text-align: center;
  padding: 15px;
  border-radius: 8px;
  margin: 10px 0;
  font-weight: bold;
  
  ${props => props.type === 'loading' && `
    background: #e3f2fd;
    color: #1565c0;
  `}
  
  ${props => props.type === 'success' && `
    background: #e8f5e8;
    color: #2e7d32;
  `}
  
  ${props => props.type === 'error' && `
    background: #ffebee;
    color: #c62828;
  `}
  
  ${props => props.type === 'info' && `
    background: #f3e5f5;
    color: #7b1fa2;
  `}
`;

const ModelInfo = styled.div`
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
  font-size: 0.9rem;
  color: #666;
`;

const CodeBlock = styled.pre`
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 15px;
  overflow-x: auto;
  font-size: 0.85rem;
  color: #333;
  margin: 10px 0;
`;

function App() {
  const [inputs, setInputs] = useState(Array(10).fill(0.5));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [showAPI, setShowAPI] = useState(false);

  const modelId = "model_1757095579212_5ee9v9o1t";

  const handleInputChange = (index, value) => {
    const newInputs = [...inputs];
    newInputs[index] = parseFloat(value) || 0;
    setInputs(newInputs);
  };

  const generateRandomInputs = () => {
    const randomInputs = Array(10).fill(0).map(() => 
      parseFloat((Math.random()).toFixed(3))
    );
    setInputs(randomInputs);
  };

  const clearInputs = () => {
    setInputs(Array(10).fill(0));
    setPrediction(null);
    setStatus(null);
  };

  const mockPredict = async () => {
    setLoading(true);
    setStatus({ type: 'loading', message: 'Running neural network prediction...' });
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    try {
      // Mock prediction response based on your model
      const mockPredictions = inputs.map((val, idx) => 
        Math.random() * 0.8 + val * 0.2
      );
      
      const maxPrediction = Math.max(...mockPredictions);
      const predictedClass = mockPredictions.indexOf(maxPrediction);
      const confidence = Math.random() * 0.3 + 0.7; // 70-100%
      
      const result = {
        success: true,
        prediction_id: `pred_${Date.now()}_demo`,
        model_id: modelId,
        predictions: mockPredictions,
        predicted_class: predictedClass,
        confidence: confidence,
        input_received: inputs,
        timestamp: new Date().toISOString()
      };
      
      setPrediction(result);
      setStatus({ 
        type: 'success', 
        message: `Prediction completed! Class ${predictedClass} with ${(confidence * 100).toFixed(1)}% confidence` 
      });
      
    } catch (error) {
      setStatus({ 
        type: 'error', 
        message: 'Prediction failed: ' + error.message 
      });
    }
    
    setLoading(false);
  };

  const chartData = prediction ? 
    prediction.predictions.map((prob, index) => ({
      class: `Class ${index}`,
      probability: prob,
      isWinner: index === prediction.predicted_class
    })) : [];

  const COLORS = ['#667eea', '#764ba2', '#28a745', '#dc3545', '#ffc107', 
                 '#17a2b8', '#6c757d', '#343a40', '#e83e8c', '#20c997'];

  return (
    <Container>
      <Header>🧠 Neural Network Predictor</Header>
      <Subtitle>
        Interactive demo of your trained neural network model<br/>
        Model ID: {modelId}
      </Subtitle>
      
      <ModelInfo>
        <strong>Model Architecture:</strong> 256→128→64→32→10 Feedforward Network<br/>
        <strong>Training Accuracy:</strong> 90.63% | <strong>Loss:</strong> 0.268 | <strong>Confidence:</strong> 92.66%<br/>
        <strong>Input:</strong> 10 floating-point numbers (0.0 - 1.0) | <strong>Output:</strong> 10-class classification
      </ModelInfo>

      <Card>
        <h2>🎛️ Input Parameters</h2>
        <p style={{color: '#666', marginBottom: '20px'}}>
          Enter 10 numerical values between 0.0 and 1.0 for your prediction:
        </p>
        
        <InputGrid>
          {inputs.map((value, index) => (
            <InputGroup key={index}>
              <Label>Input {index + 1}</Label>
              <Input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={value}
                onChange={(e) => handleInputChange(index, e.target.value)}
                placeholder="0.00"
              />
            </InputGroup>
          ))}
        </InputGrid>

        <ButtonGroup>
          <PredictButton onClick={mockPredict} disabled={loading}>
            {loading ? '🔄 Predicting...' : '🚀 Mock Predict'}
          </PredictButton>
          <RandomButton onClick={generateRandomInputs}>
            🎲 Random Values
          </RandomButton>
          <ClearButton onClick={clearInputs}>
            🗑️ Clear All
          </ClearButton>
          <APIButton onClick={() => setShowAPI(!showAPI)}>
            📚 {showAPI ? 'Hide' : 'Show'} API Code
          </APIButton>
        </ButtonGroup>

        {status && (
          <StatusMessage type={status.type}>
            {status.message}
          </StatusMessage>
        )}

        {showAPI && (
          <div>
            <h3>🔗 Real API Integration</h3>
            <p>To connect to your actual neural network model, replace the mock function with:</p>
            <CodeBlock>
{`// Real API call to Flow Nexus
const realPredict = async () => {
  try {
    const response = await fetch('/api/neural/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_id: "${modelId}",
        input: inputs
      })
    });
    
    const result = await response.json();
    setPrediction(result);
  } catch (error) {
    console.error('Prediction failed:', error);
  }
};

// Or using MCP client:
const mcpPredict = async () => {
  const result = await mcp.call("mcp__flow-nexus__neural_predict", {
    model_id: "${modelId}",
    input: inputs
  });
  setPrediction(result);
};`}
            </CodeBlock>
          </div>
        )}
      </Card>

      <ResultCard show={prediction}>
        <h2>📊 Prediction Results</h2>
        
        {prediction && (
          <>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap'}}>
              <div>
                <h3 style={{margin: 0, color: '#667eea'}}>
                  🎯 Predicted Class: {prediction.predicted_class}
                </h3>
                <p style={{margin: '5px 0', color: '#666'}}>
                  Probability: {(prediction.predictions[prediction.predicted_class] * 100).toFixed(2)}%
                </p>
              </div>
              <div style={{textAlign: 'right'}}>
                <p style={{margin: 0, color: '#666'}}>Overall Confidence:</p>
                <h3 style={{margin: 0, color: '#28a745'}}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </h3>
              </div>
            </div>

            <h3>📈 Class Probability Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="class" />
                <YAxis />
                <Tooltip 
                  formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Probability']}
                />
                <Bar dataKey="probability">
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.isWinner ? '#28a745' : COLORS[index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div style={{marginTop: '20px', fontSize: '0.9rem', color: '#666'}}>
              <p><strong>Prediction ID:</strong> {prediction.prediction_id}</p>
              <p><strong>Input Values:</strong> [{prediction.input_received.join(', ')}]</p>
              <p><strong>Timestamp:</strong> {new Date(prediction.timestamp).toLocaleString()}</p>
            </div>
          </>
        )}
      </ResultCard>

      <Card>
        <h2>ℹ️ About This App</h2>
        <p>This is a demo React application that interfaces with your trained neural network model. Features include:</p>
        <ul>
          <li>🎛️ Interactive 10-parameter input interface</li>
          <li>📊 Real-time prediction visualization with bar charts</li>
          <li>🎲 Random value generation for testing</li>
          <li>📚 API integration examples</li>
          <li>📱 Responsive design for mobile and desktop</li>
          <li>🎨 Modern UI with styled-components</li>
        </ul>
        <p><strong>Note:</strong> Currently using mock predictions. Replace with real API calls to use your actual model.</p>
      </Card>
    </Container>
  );
}

export default App;