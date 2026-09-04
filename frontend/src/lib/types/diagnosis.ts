/**
 * TypeScript Type Definitions Synchronized Exactly with Phase 9 FastAPI Pydantic Models.
 */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  git_sha: string;
  timestamp: string;
}

export interface ReadinessResponse {
  ready: boolean;
  status: string;
  components: Record<string, boolean>;
  timestamp: string;
}

export interface SensorTelemetryInput {
  temperature?: number;
  temperature_unit?: string;
  vibration?: number;
  vibration_unit?: string;
  rpm?: number;
  current?: number;
  current_unit?: string;
  pressure?: number;
  pressure_unit?: string;
  custom_parameters?: Record<string, number>;
}

export interface EquipmentMetadataInput {
  equipment_type: string;
  manufacturer?: string;
  model?: string;
  serial_number?: string;
  operating_mode?: string;
}

export interface EvidenceItemResponse {
  evidence_id: string;
  category: string;
  source_modality: string;
  description: string;
  confidence: float_number;
  raw_measurement?: string;
  iso_standard_reference?: string;
  document_reference?: string;
  page_number?: number;
  is_verified_citation: boolean;
}

type float_number = number;

export interface ConfidenceDecompositionResponse {
  overall_confidence: number;
  multimodal_agreement: string;
  sensor_evidence_strength: string;
  acoustic_evidence_strength: string;
  visual_evidence_strength: string;
  technical_knowledge_match: string;
  contradiction_penalty: number;
  rationale_summary: string;
}

export interface PrimaryDiagnosisResponse {
  primary_diagnosis: string;
  diagnostic_confidence: number;
  severity: string;
  confidence_decomposition: ConfidenceDecompositionResponse;
}

export interface RecommendedActionResponse {
  action_id: string;
  action_type: string;
  description: string;
  target_component: string;
  justifying_evidence_ids: string[];
  citation?: string;
}

export interface DiagnosisResponse {
  case_id: string;
  request_id: string;
  timestamp: string;
  status: string;
  equipment: Record<string, any>;
  problem_summary: string;
  available_modalities: string[];
  diagnosis: PrimaryDiagnosisResponse;
  evidence_inventory: EvidenceItemResponse[];
  claim_mappings: any[];
  alternative_hypotheses: any[];
  recommended_actions: RecommendedActionResponse[];
  uncertainty_profile: Record<string, any>;
  unsupported_claims: string[];
  audit_summary: Record<string, any>;
  markdown_report: string;
}

export interface RAGQueryMatch {
  chunk_id: string;
  document_name: string;
  page_number?: number;
  section?: string;
  similarity_score: number;
  content_preview: string;
}

export interface RAGQueryResponse {
  query: string;
  results_count: number;
  matches: RAGQueryMatch[];
  timestamp: string;
}
