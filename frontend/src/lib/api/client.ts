/**
 * Centralized API Client interfacing with FastAPI Backend.
 */

import {
  DiagnosisResponse,
  HealthResponse,
  ReadinessResponse,
  RAGQueryResponse,
  SensorTelemetryInput,
  EquipmentMetadataInput,
} from "@/lib/types/diagnosis";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export class APIClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async checkHealth(): Promise<HealthResponse> {
    const res = await fetch(`${this.baseUrl}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    return res.json();
  }

  async checkReadiness(): Promise<ReadinessResponse> {
    const res = await fetch(`${this.baseUrl}/ready`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new Error(`Readiness probe failed (${res.status})`);
    return res.json();
  }

  async queryKnowledge(q: string, equipmentType?: string, topK: number = 5): Promise<RAGQueryResponse> {
    const params = new URLSearchParams({ q, top_k: topK.toString() });
    if (equipmentType) params.append("equipment_type", equipmentType);

    const res = await fetch(`${this.baseUrl}/api/v1/knowledge/query?${params.toString()}`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new Error(`Knowledge query failed (${res.status})`);
    return res.json();
  }

  async submitDiagnosis(
    technicianDescription?: string,
    imageFile?: File,
    audioFile?: File,
    sensorData?: SensorTelemetryInput,
    equipmentMetadata?: EquipmentMetadataInput
  ): Promise<DiagnosisResponse> {
    const formData = new FormData();

    if (technicianDescription) {
      formData.append("technician_description", technicianDescription);
    }
    if (imageFile) {
      formData.append("image", imageFile);
    }
    if (audioFile) {
      formData.append("audio", audioFile);
    }
    if (sensorData) {
      formData.append("sensor_json", JSON.stringify(sensorData));
    }
    if (equipmentMetadata) {
      formData.append("equipment_json", JSON.stringify(equipmentMetadata));
    }

    const res = await fetch(`${this.baseUrl}/api/v1/diagnose`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      const msg = errJson.message || `Diagnosis failed with status ${res.status}`;
      throw new Error(msg);
    }

    return res.json();
  }
}

export const apiClient = new APIClient();
