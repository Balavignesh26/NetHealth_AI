# Demo Script

**Goal**: Demonstrate the "Industry-Grade" Observability Platform.

## 1. The Hook (Top Bar)
*   **Action**: Open the dashboard (Default: "Normal Operation").
*   **Say**: "This is Belden ONE View. At a glance, you see the system health is 100/100. No alerts."
*   **Show**: Point to the Green "Healthy" status top right.

## 2. The Foundation (Topology)
*   **Action**: Scroll to "Network Map".
*   **Say**: "We don't just see list of devices. We understand the physical topology. Core -> Edge -> PLCs."
*   **Show**: The interactive graph.

## 3. The Problem (Inject Fault)
*   **Action**: In the Sidebar, select **"Inject Fault (Cable Failure)"** and click **"Run Analysis"**.
*   **Say**: "Let's simulate a real-world issue. A cable failure at the Edge Switch."
*   **Observation**:
    *   Top status turns **RED**.
    *   ONE Score drops (e.g., to ~70).
    *   Topology nodes turn Red.

## 4. The AI Solution (Insights)
*   **Action**: Look at the "AI Root Cause Analysis" panel on the right.
*   **Say**: "Normally, you'd see alerts everywhere—Packet Loss on PLCs, Latency on HMIs. It's a storm."
*   **Say**: "But our AI correlates these. Look here."
*   **Read**: "Root Cause: **Edge Switch A**. Confidence: **Very High**."
*   **Show**: The recommendation: "Check physical cabling."

## 5. The Conclusion
*   **Say**: "We turned a flood of alerts into a single, actionable insight. 4 hours of troubleshooting reduced to 4 seconds."
