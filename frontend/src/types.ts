export interface Project {
  id: string;
  name: string;
  path: string;
  created_at: string;
  last_opened_at: string;
  description: string;
  file_count?: number;
  exists?: boolean;
}

export interface WorkspaceFile {
  name: string;
  path: string;
  size_bytes: number;
  size_kb: number;
  modified: string;
  ext: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
}

export interface PlanStep {
  step_num: number;
  action_type: string;
  description: string;
  status: string;
}

export interface ExecutedStep {
  step_num: number;
  action_type: string;
  description: string;
  tool_name: string;
  tool_args: Record<string, any>;
  observation: string;
  success: boolean;
}

export interface VerificationInfo {
  passed: boolean;
  status: string;
  diagnostics: string;
  suggested_fix?: string | null;
}

export interface AgentTurnResponse {
  user_message: string;
  response: string;
  intent: {
    name: string;
    confidence: number;
    explanation: string;
  };
  entities: {
    language: string;
    filename?: string;
    purpose?: string;
    inputs?: any[];
    run_after?: boolean;
  };
  plan: {
    goal: string;
    state: string;
    steps: PlanStep[];
  };
  executed_steps: ExecutedStep[];
  verification?: VerificationInfo | null;
  active_file?: string;
  terminal_output?: string;
  timestamp: string;
}
