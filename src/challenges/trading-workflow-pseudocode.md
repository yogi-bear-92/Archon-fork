# SPARC Phase 2: PSEUDOCODE DESIGN
## Trading Workflow Algorithm with pgmq Integration

### High-Level Algorithm Flow:

```pseudocode
FUNCTION trading_workflow_processor()
    INITIALIZE queue_connection to 'workflow_main_queue'
    INITIALIZE results_queue to 'workflow_results'
    
    WHILE system_active:
        // Step 1: Read signals from queue (non-blocking)
        message = READ_FROM_QUEUE('workflow_main_queue', timeout=30, limit=1)
        
        IF message IS NOT NULL:
            // Step 2: Extract trading data
            rsi_value = EXTRACT(message.payload, 'rsi')
            price = EXTRACT(message.payload, 'price')
            timestamp = CURRENT_TIMESTAMP()
            
            // Step 3: Apply RSI trading logic
            signal = CALCULATE_TRADING_SIGNAL(rsi_value)
            
            // Step 4: Create workflow execution record
            execution_id = LOG_WORKFLOW_EXECUTION(
                workflow_type: 'trading_signal',
                input_data: message.payload,
                status: 'processing'
            )
            
            // Step 5: Submit result to results queue
            result_payload = CREATE_RESULT_PAYLOAD(
                execution_id: execution_id,
                signal: signal,
                rsi: rsi_value,
                price: price,
                confidence: CALCULATE_CONFIDENCE(rsi_value)
            )
            
            SEND_TO_QUEUE('workflow_results', result_payload)
            
            // Step 6: Update execution status
            UPDATE_WORKFLOW_EXECUTION(execution_id, 'completed')
            
        ELSE:
            // No messages, brief pause to prevent CPU spinning
            SLEEP(1000ms)
    END WHILE
END FUNCTION

FUNCTION calculate_trading_signal(rsi_value INTEGER)
    IF rsi_value < 30:
        RETURN 'BUY'
    ELSIF rsi_value > 70:
        RETURN 'SELL'
    ELSE:
        RETURN 'HOLD'
    END IF
END FUNCTION

FUNCTION calculate_confidence(rsi_value INTEGER)
    // Distance from neutral zone (30-70)
    IF rsi_value < 30:
        confidence = (30 - rsi_value) / 30.0  // 0-1 scale
    ELSIF rsi_value > 70:
        confidence = (rsi_value - 70) / 30.0   // 0-1 scale
    ELSE:
        confidence = 0.1  // Low confidence for HOLD
    END IF
    
    RETURN MIN(confidence, 1.0)
END FUNCTION
```

### PostgreSQL Function Design:

```pseudocode
-- Core trading signal function (from starter code, enhanced)
CREATE FUNCTION enhanced_trade_signal(
    rsi INTEGER,
    price DECIMAL DEFAULT NULL,
    volume DECIMAL DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    signal TEXT;
    confidence DECIMAL;
    risk_level TEXT;
BEGIN
    // Main RSI logic
    IF rsi < 30 THEN 
        signal := 'BUY';
        confidence := LEAST((30 - rsi) / 30.0, 1.0);
        risk_level := CASE WHEN rsi < 20 THEN 'HIGH' ELSE 'MEDIUM' END;
    ELSIF rsi > 70 THEN 
        signal := 'SELL';
        confidence := LEAST((rsi - 70) / 30.0, 1.0);
        risk_level := CASE WHEN rsi > 80 THEN 'HIGH' ELSE 'MEDIUM' END;
    ELSE 
        signal := 'HOLD';
        confidence := 0.1;
        risk_level := 'LOW';
    END IF;
    
    // Return structured result
    RETURN jsonb_build_object(
        'signal', signal,
        'confidence', confidence,
        'risk_level', risk_level,
        'rsi', rsi,
        'price', COALESCE(price, 0),
        'timestamp', EXTRACT(EPOCH FROM NOW()),
        'strategy', 'rsi_basic'
    );
END;
$$ LANGUAGE plpgsql;
```

### Queue Processing Pattern:

```pseudocode
-- Queue processor function
CREATE FUNCTION process_trading_queue() RETURNS INTEGER AS $$
DECLARE
    msg_record RECORD;
    result_data JSONB;
    execution_id UUID;
    processed_count INTEGER := 0;
BEGIN
    // Read messages in batches for efficiency
    FOR msg_record IN 
        SELECT * FROM pgmq.read('workflow_main_queue', 30, 10)
    LOOP
        // Extract message data
        result_data := enhanced_trade_signal(
            (msg_record.message->>'rsi')::INTEGER,
            (msg_record.message->>'price')::DECIMAL
        );
        
        // Log execution
        INSERT INTO workflow_executions (
            id, workflow_type, input_data, 
            output_data, status, created_at
        ) VALUES (
            gen_random_uuid(),
            'trading_signal',
            msg_record.message,
            result_data,
            'completed',
            NOW()
        ) RETURNING id INTO execution_id;
        
        // Send result to output queue
        PERFORM pgmq.send(
            'workflow_results',
            jsonb_build_object(
                'execution_id', execution_id,
                'original_msg_id', msg_record.msg_id,
                'result', result_data,
                'processed_at', NOW()
            )
        );
        
        // Archive processed message
        PERFORM pgmq.archive('workflow_main_queue', msg_record.msg_id);
        
        processed_count := processed_count + 1;
    END LOOP;
    
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;
```

### Error Handling & Resilience:

```pseudocode
CREATE FUNCTION safe_trading_processor() RETURNS INTEGER AS $$
DECLARE
    result INTEGER;
BEGIN
    BEGIN
        result := process_trading_queue();
        
        // Log successful processing
        INSERT INTO processing_log (
            timestamp, status, messages_processed, error_message
        ) VALUES (
            NOW(), 'SUCCESS', result, NULL
        );
        
    EXCEPTION 
        WHEN OTHERS THEN
            // Log error
            INSERT INTO processing_log (
                timestamp, status, messages_processed, error_message
            ) VALUES (
                NOW(), 'ERROR', 0, SQLERRM
            );
            
            // Re-raise for monitoring
            RAISE;
    END;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
```

### Performance Optimization Patterns:
1. **Batch Processing**: Read multiple messages per iteration
2. **Connection Pooling**: Reuse database connections
3. **Streaming Results**: Process and forward without accumulation
4. **Error Recovery**: Graceful handling of malformed messages
5. **Monitoring**: Execution logging for performance tracking

### Memory-Efficient Implementation:
- Process messages in small batches (10-50)
- Use streaming operations for large datasets
- Minimal state retention between cycles
- Efficient JSON processing with native PostgreSQL functions