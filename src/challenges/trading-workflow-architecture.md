# SPARC Phase 3: ARCHITECTURE DESIGN
## Trading Workflow System with pgmq Integration

### System Architecture Overview:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLOW NEXUS TRADING WORKFLOW                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   Data Ingress  │───▶│  Signal Processor│───▶│ Result Handler  │  │
│  │                 │    │                 │    │                 │  │
│  │ • Market Data   │    │ • RSI Analysis  │    │ • Queue Output  │  │
│  │ • External APIs │    │ • Risk Scoring  │    │ • Audit Trail   │  │
│  │ • Manual Signals│    │ • Confidence    │    │ • Monitoring    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           │                       │                       │         │
│           ▼                       ▼                       ▼         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ workflow_main   │    │   PostgreSQL    │    │ workflow_results│  │
│  │     _queue      │    │   Functions     │    │     _queue      │  │
│  │                 │    │                 │    │                 │  │
│  │ pgmq.send()     │    │ • trade_signal()│    │ pgmq.read()     │  │
│  │ pgmq.read()     │    │ • process_queue │    │ pgmq.archive()  │  │
│  └─────────────────┘    │ • error_handler │    └─────────────────┘  │
│                         └─────────────────┘                         │
│                                  │                                  │
│                                  ▼                                  │
│                         ┌─────────────────┐                         │
│                         │workflow_executions                        │
│                         │                 │                         │
│                         │ • execution_id  │                         │
│                         │ • input_data    │                         │
│                         │ • output_data   │                         │
│                         │ • status        │                         │
│                         │ • timestamps    │                         │
│                         └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Architecture:

#### 1. **Queue Layer (pgmq)**
```sql
-- Input Queue Configuration
CREATE TABLE IF NOT EXISTS pgmq.workflow_main_queue (
    msg_id BIGINT PRIMARY KEY,
    read_ct INTEGER NOT NULL DEFAULT 0,
    enqueued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    vt TIMESTAMP WITH TIME ZONE NOT NULL,
    message JSONB NOT NULL
);

-- Output Queue Configuration  
CREATE TABLE IF NOT EXISTS pgmq.workflow_results (
    msg_id BIGINT PRIMARY KEY,
    read_ct INTEGER NOT NULL DEFAULT 0,
    enqueued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    vt TIMESTAMP WITH TIME ZONE NOT NULL,
    message JSONB NOT NULL
);

-- Dead Letter Queue for Error Handling
CREATE TABLE IF NOT EXISTS pgmq.workflow_errors (
    msg_id BIGINT PRIMARY KEY,
    read_ct INTEGER NOT NULL DEFAULT 0,
    enqueued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    vt TIMESTAMP WITH TIME ZONE NOT NULL,
    message JSONB NOT NULL,
    error_details JSONB
);
```

#### 2. **Processing Functions Layer**
```sql
-- Enhanced Trading Signal Function
CREATE OR REPLACE FUNCTION enhanced_trade_signal(
    rsi INTEGER,
    price DECIMAL DEFAULT NULL,
    volume DECIMAL DEFAULT NULL,
    market_context JSONB DEFAULT '{}'
) RETURNS JSONB AS $$
DECLARE
    signal TEXT;
    confidence DECIMAL;
    risk_level TEXT;
    market_sentiment TEXT;
    trade_strength DECIMAL;
BEGIN
    -- Core RSI Logic with Market Context
    IF rsi < 30 THEN 
        signal := 'BUY';
        confidence := LEAST((30 - rsi) / 30.0, 1.0);
        risk_level := CASE WHEN rsi < 20 THEN 'HIGH' ELSE 'MEDIUM' END;
        trade_strength := CASE WHEN rsi < 20 THEN 0.9 ELSE 0.7 END;
    ELSIF rsi > 70 THEN 
        signal := 'SELL';
        confidence := LEAST((rsi - 70) / 30.0, 1.0);
        risk_level := CASE WHEN rsi > 80 THEN 'HIGH' ELSE 'MEDIUM' END;
        trade_strength := CASE WHEN rsi > 80 THEN 0.9 ELSE 0.7 END;
    ELSE 
        signal := 'HOLD';
        confidence := 0.1;
        risk_level := 'LOW';
        trade_strength := 0.1;
    END IF;
    
    -- Market Sentiment Analysis
    market_sentiment := CASE 
        WHEN (market_context->>'trend')::TEXT = 'bullish' AND signal = 'BUY' THEN 'STRONG_BUY'
        WHEN (market_context->>'trend')::TEXT = 'bearish' AND signal = 'SELL' THEN 'STRONG_SELL'
        ELSE signal
    END;
    
    -- Return Enhanced Signal
    RETURN jsonb_build_object(
        'signal', signal,
        'enhanced_signal', market_sentiment,
        'confidence', confidence,
        'trade_strength', trade_strength,
        'risk_level', risk_level,
        'rsi', rsi,
        'price', COALESCE(price, 0),
        'volume', COALESCE(volume, 0),
        'market_context', market_context,
        'timestamp', EXTRACT(EPOCH FROM NOW()),
        'strategy', 'enhanced_rsi',
        'version', '1.0'
    );
END;
$$ LANGUAGE plpgsql;
```

#### 3. **Workflow Execution Layer**
```sql
-- Workflow Executions Table (Enhanced)
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type VARCHAR(50) NOT NULL,
    workflow_version VARCHAR(10) DEFAULT '1.0',
    input_data JSONB NOT NULL,
    output_data JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    error_details JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS workflow_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID REFERENCES workflow_executions(id),
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL NOT NULL,
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 4. **Main Processing Engine**
```sql
CREATE OR REPLACE FUNCTION process_trading_workflow(
    batch_size INTEGER DEFAULT 10,
    timeout_seconds INTEGER DEFAULT 30
) RETURNS JSONB AS $$
DECLARE
    msg_record RECORD;
    result_data JSONB;
    execution_id UUID;
    processed_count INTEGER := 0;
    error_count INTEGER := 0;
    start_time TIMESTAMP;
    processing_stats JSONB;
BEGIN
    start_time := NOW();
    
    -- Process messages in batches
    FOR msg_record IN 
        SELECT * FROM pgmq.read('workflow_main_queue', timeout_seconds, batch_size)
    LOOP
        BEGIN
            -- Create workflow execution record
            INSERT INTO workflow_executions (
                workflow_type, input_data, status, started_at
            ) VALUES (
                'trading_signal', msg_record.message, 'processing', NOW()
            ) RETURNING id INTO execution_id;
            
            -- Process trading signal
            result_data := enhanced_trade_signal(
                (msg_record.message->>'rsi')::INTEGER,
                (msg_record.message->>'price')::DECIMAL,
                (msg_record.message->>'volume')::DECIMAL,
                COALESCE(msg_record.message->'market_context', '{}')
            );
            
            -- Update execution with results
            UPDATE workflow_executions SET
                output_data = result_data,
                status = 'completed',
                completed_at = NOW(),
                processing_time_ms = EXTRACT(MILLISECONDS FROM NOW() - started_at)
            WHERE id = execution_id;
            
            -- Send to results queue
            PERFORM pgmq.send(
                'workflow_results',
                jsonb_build_object(
                    'execution_id', execution_id,
                    'original_msg_id', msg_record.msg_id,
                    'result', result_data,
                    'processing_time', EXTRACT(MILLISECONDS FROM NOW() - start_time),
                    'processed_at', NOW()
                )
            );
            
            -- Archive processed message
            PERFORM pgmq.archive('workflow_main_queue', msg_record.msg_id);
            
            processed_count := processed_count + 1;
            
        EXCEPTION WHEN OTHERS THEN
            error_count := error_count + 1;
            
            -- Log error
            UPDATE workflow_executions SET
                status = 'error',
                error_details = jsonb_build_object(
                    'error_code', SQLSTATE,
                    'error_message', SQLERRM,
                    'error_timestamp', NOW()
                ),
                completed_at = NOW()
            WHERE id = execution_id;
            
            -- Send to error queue
            PERFORM pgmq.send(
                'workflow_errors',
                jsonb_build_object(
                    'original_message', msg_record.message,
                    'error_details', jsonb_build_object(
                        'error_code', SQLSTATE,
                        'error_message', SQLERRM
                    ),
                    'failed_at', NOW()
                )
            );
        END;
    END LOOP;
    
    -- Compile processing statistics
    processing_stats := jsonb_build_object(
        'processed_count', processed_count,
        'error_count', error_count,
        'success_rate', CASE 
            WHEN processed_count + error_count = 0 THEN 0 
            ELSE processed_count::DECIMAL / (processed_count + error_count) 
        END,
        'processing_time_ms', EXTRACT(MILLISECONDS FROM NOW() - start_time),
        'batch_size', batch_size,
        'timestamp', NOW()
    );
    
    RETURN processing_stats;
END;
$$ LANGUAGE plpgsql;
```

### 5. **Monitoring and Analytics Layer**
```sql
-- Performance Dashboard View
CREATE OR REPLACE VIEW trading_workflow_dashboard AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total_executions,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
    COUNT(CASE WHEN status = 'error' THEN 1 END) as failed,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN (output_data->>'signal') = 'BUY' THEN 1 END) as buy_signals,
    COUNT(CASE WHEN (output_data->>'signal') = 'SELL' THEN 1 END) as sell_signals,
    COUNT(CASE WHEN (output_data->>'signal') = 'HOLD' THEN 1 END) as hold_signals
FROM workflow_executions
WHERE workflow_type = 'trading_signal'
    AND created_at >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;
```

### Deployment Architecture:

#### **Memory-Optimized Configuration:**
- **Connection Pooling**: Max 5 concurrent connections
- **Batch Processing**: 10 messages per batch to prevent memory overflow  
- **Streaming Operations**: No large data accumulation
- **Error Recovery**: Graceful degradation under memory pressure

#### **Scalability Patterns:**
- **Horizontal Scaling**: Multiple processor instances
- **Queue Partitioning**: Separate queues for different trading pairs
- **Result Aggregation**: Consolidated analytics processing
- **Load Balancing**: Round-robin message distribution

#### **Integration Points:**
1. **Flow Nexus API**: RESTful endpoints for workflow management
2. **Real-time Monitoring**: WebSocket connections for live updates
3. **External Data Sources**: Market data feeds and indicators
4. **Alert System**: Notification triggers for significant signals

### Performance Targets:
- **Throughput**: 100+ signals/second
- **Latency**: <50ms per signal processing
- **Availability**: 99.9% uptime
- **Memory Usage**: <100MB under normal load
- **Error Rate**: <1% processing failures