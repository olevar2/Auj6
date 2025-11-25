# AUJ Platform Architecture Diagrams

## 1. Component Architecture Diagram

```mermaid
graph TB
    subgraph Core ["Core System"]
        GAC[Genius Agent Coordinator]
        PM[Performance Monitor]
        CM[Configuration Management]
        MS[Messaging System]
    end

    subgraph Agents ["Agent Hierarchy"]
        subgraph Alpha["Alpha Tier"]
            SE[StrategyExpert]
            RG[RiskGenius]
        end
        
        subgraph Beta["Beta Tier"]
            PM2[PatternMaster]
            PS[PairSpecialist]
            SE2[SessionExpert]
            IE[IndicatorExpert]
        end
        
        subgraph Gamma["Gamma Tier"]
            EE[ExecutionExpert]
            DM[DecisionMaster]
            MA[MicrostructureAgent]
            SiE[SimulationExpert]
        end
    end

    subgraph Infrastructure ["Infrastructure"]
        DB[(Postgres DB)]
        MQ[RabbitMQ]
        PRO[Prometheus]
        GRA[Grafana]
        NGX[Nginx]
    end

    subgraph Execution ["Execution Layer"]
        EH[Execution Handler]
        RM[Risk Manager]
        DMT[Deal Monitoring]
    end

    subgraph Analysis ["Analysis Components"]
        IA[Indicator Analysis]
        MA2[Market Analysis]
        VA[Validation Analysis]
    end

    GAC --> |Orchestrates| Agents
    GAC --> |Controls| Analysis
    GAC --> |Validates| Execution
    
    MS --> |Messages| Agents
    MS --> |Events| EH
    MS --> |Metrics| PM
    
    PM --> PRO
    PRO --> GRA
    
    EH --> DB
    EH --> MQ
    
    Analysis --> |Data| Agents
    Agents --> |Decisions| GAC
    GAC --> |Signals| EH
    
    EH --> |Execution| RM
    RM --> |Monitoring| DMT

## 2. Process Flow Diagram

```mermaid
sequenceDiagram
    participant MD as Market Data
    participant IE as Indicator Engine
    participant GC as Genius Coordinator
    participant AG as Agent Hierarchy
    participant EH as Execution Handler
    participant RM as Risk Manager
    
    Note over MD,RM: Complete Analysis Cycle (25-35s)
    
    MD->>IE: Collect Market Data (2-3s)
    activate IE
    
    IE->>GC: Process Indicators (8-12s)
    activate GC
    
    GC->>AG: Initiate Analysis
    
    par Parallel Agent Analysis (8-15s)
        AG->>AG: Alpha Tier Analysis
        Note over AG: StrategyExpert & RiskGenius
        
        AG->>AG: Beta Tier Analysis
        Note over AG: PatternMaster, PairSpecialist, SessionExpert
        
        AG->>AG: Gamma Tier Analysis
        Note over AG: ExecutionExpert, DecisionMaster
    end
    
    AG->>GC: Agent Decisions
    
    GC->>GC: Decision Synthesis (3-5s)
    Note over GC: Hierarchical Consensus
    
    GC->>EH: Trade Signal
    deactivate GC
    
    EH->>RM: Validate Risk (2-3s)
    activate RM
    
    RM->>EH: Risk Approval
    deactivate RM
    
    EH->>EH: Execute Trade
    deactivate IE
    
    Note over MD,RM: Continuous Monitoring
```

## Diagrams Legend

### Component Architecture
- **Core System**: Central platform orchestration and management
- **Agent Hierarchy**: Three-tier agent system (Alpha, Beta, Gamma)
- **Infrastructure**: Supporting services and databases
- **Execution Layer**: Trade execution and monitoring
- **Analysis Components**: Market and indicator analysis systems

### Process Flow
- Timing for each phase shown in parentheses
- Parallel processing sections shown in 'par' blocks
- Activation/deactivation of components shown with bars
- Notes indicate important process details
