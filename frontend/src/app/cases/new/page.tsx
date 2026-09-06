"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { 
  Upload, 
  Image as ImageIcon, 
  Volume2, 
  Cpu, 
  FileText, 
  CheckCircle2, 
  AlertTriangle, 
  Loader2, 
  Trash2,
  HelpCircle,
  Sparkles,
  ArrowRight
} from "lucide-react";
import { apiClient } from "@/lib/api/client";
import { SensorTelemetryInput, EquipmentMetadataInput } from "@/lib/types/diagnosis";
import { formatFileSize } from "@/lib/utils";

export default function NewCasePage() {
  const router = useRouter();

  // 1. Equipment Metadata State
  const [equipmentType, setEquipmentType] = useState("motor");
  const [manufacturer, setManufacturer] = useState("Siemens");
  const [model, setModel] = useState("M-4500");
  const [serialNumber, setSerialNumber] = useState("SN-94821");

  // 2. Problem Description State
  const [description, setDescription] = useState(
    "High frequency squealing noise originating from drive-end bearing. Casing temperature elevated to 88 degC."
  );

  // 3. Media Upload State
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);

  // 4. Sensor Telemetry State
  const [temperature, setTemperature] = useState<string>("88.0");
  const [vibration, setVibration] = useState<string>("7.4");
  const [rpm, setRpm] = useState<string>("1490");
  const [current, setCurrent] = useState<string>("8.5");
  const [pressure, setPressure] = useState<string>("1.2");

  // 5. Submission & Loading State
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setIsSubmitting(true);

    try {
      const sensorData: SensorTelemetryInput = {
        temperature: temperature ? parseFloat(temperature) : undefined,
        vibration: vibration ? parseFloat(vibration) : undefined,
        rpm: rpm ? parseFloat(rpm) : undefined,
        current: current ? parseFloat(current) : undefined,
        pressure: pressure ? parseFloat(pressure) : undefined,
      };

      const eqMetadata: EquipmentMetadataInput = {
        equipment_type: equipmentType,
        manufacturer: manufacturer || undefined,
        model: model || undefined,
        serial_number: serialNumber || undefined,
      };

      const result = await apiClient.submitDiagnosis(
        description || undefined,
        imageFile || undefined,
        audioFile || undefined,
        sensorData,
        eqMetadata
      );

      // Store response in sessionStorage and navigate to result
      sessionStorage.setItem(`case_${result.case_id}`, JSON.stringify(result));
      router.push(`/cases/${result.case_id}`);
    } catch (err: any) {
      setErrorMessage(err.message || "An unexpected error occurred during diagnosis.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center space-x-1.5 bg-black/[0.04] text-[#6e6e73] px-3 py-1 rounded-full text-xs font-medium">
          <Sparkles className="w-3 h-3 text-[#1d1d1f]" />
          <span>New Inspection Submission</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-[#1d1d1f]">
          Create Diagnostic Case
        </h1>
        <p className="text-sm text-[#6e6e73] max-w-lg mx-auto leading-relaxed">
          Provide asset parameters, physical sensor telemetry, technician observations, and upload image or acoustic samples.
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 bg-white border border-[#ff3b30]/30 rounded-[18px] flex items-start space-x-3 text-[#ff3b30] text-sm shadow-[0_2px_12px_rgba(255,59,48,0.06)]">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 text-[#ff3b30] mt-0.5" />
          <div>
            <div className="font-semibold">Submission Incomplete</div>
            <div className="text-xs text-[#6e6e73] mt-0.5">{errorMessage}</div>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Section 1: Equipment Profile */}
        <div className="bg-white p-6 sm:p-8 rounded-[24px] border border-black/[0.06] shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-black/[0.04]">
            <h2 className="text-base font-semibold text-[#1d1d1f] flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-[#1d1d1f]" />
              <span>1. Equipment Metadata</span>
            </h2>
            <span className="text-[11px] text-[#86868b]">Required for OEM matching</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
            <div>
              <label className="block text-xs font-medium text-[#6e6e73] mb-1.5">Equipment Type *</label>
              <select
                value={equipmentType}
                onChange={(e) => setEquipmentType(e.target.value)}
                className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[14px] px-3.5 py-2.5 text-[#1d1d1f] text-sm transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none"
              >
                <option value="motor">Electric Motor</option>
                <option value="pump">Centrifugal Pump</option>
                <option value="gearbox">Industrial Gearbox</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-[#6e6e73] mb-1.5">Manufacturer</label>
              <input
                type="text"
                value={manufacturer}
                onChange={(e) => setManufacturer(e.target.value)}
                className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[14px] px-3.5 py-2.5 text-sm text-[#1d1d1f] transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none"
                placeholder="e.g. Siemens"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#6e6e73] mb-1.5">Model Identifier</label>
              <input
                type="text"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[14px] px-3.5 py-2.5 text-sm text-[#1d1d1f] transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none"
                placeholder="e.g. M-4500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#6e6e73] mb-1.5">Asset Serial Number</label>
              <input
                type="text"
                value={serialNumber}
                onChange={(e) => setSerialNumber(e.target.value)}
                className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[14px] px-3.5 py-2.5 text-sm text-[#1d1d1f] transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none"
                placeholder="e.g. SN-94821"
              />
            </div>
          </div>
        </div>

        {/* Section 2: Technician Notes */}
        <div className="bg-white p-6 sm:p-8 rounded-[24px] border border-black/[0.06] shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-black/[0.04]">
            <h2 className="text-base font-semibold text-[#1d1d1f] flex items-center space-x-2">
              <FileText className="w-4 h-4 text-[#1d1d1f]" />
              <span>2. Problem Description & Observations</span>
            </h2>
            <span className="text-[11px] text-[#86868b]">Field technician logs</span>
          </div>

          <div>
            <textarea
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe symptoms, operating sounds, or visual abnormalities..."
              className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[16px] p-3.5 text-sm text-[#1d1d1f] transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none leading-relaxed"
            />
          </div>
        </div>

        {/* Section 3: Sensor Readings */}
        <div className="bg-white p-6 sm:p-8 rounded-[24px] border border-black/[0.06] shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-black/[0.04]">
            <h2 className="text-base font-semibold text-[#1d1d1f] flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-[#1d1d1f]" />
              <span>3. Physical Sensor Telemetry</span>
            </h2>
            <span className="text-[11px] text-[#86868b]">ISO 10816 evaluation</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
            <div>
              <label className="block text-xs font-medium text-[#6e6e73] mb-1.5">Temperature (°C)</label>
              <input
                type="number"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[14px] px-3.5 py-2.5 text-sm text-[#1d1d1f] transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none"
                placeholder="e.g. 88.0"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#6e6e73] mb-1.5">Vibration (mm/s RMS)</label>
              <input
                type="number"
                step="0.1"
                value={vibration}
                onChange={(e) => setVibration(e.target.value)}
                className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[14px] px-3.5 py-2.5 text-sm text-[#1d1d1f] transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none"
                placeholder="e.g. 7.4"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#6e6e73] mb-1.5">Operating Speed (RPM)</label>
              <input
                type="number"
                value={rpm}
                onChange={(e) => setRpm(e.target.value)}
                className="w-full bg-[#f5f5f7] hover:bg-[#efeff2] border border-black/[0.06] rounded-[14px] px-3.5 py-2.5 text-sm text-[#1d1d1f] transition-all focus:border-black focus:bg-white focus:ring-4 focus:ring-black/[0.04] focus:outline-none"
                placeholder="e.g. 1490"
              />
            </div>
          </div>
        </div>

        {/* Section 4: Visual & Audio Media Uploads */}
        <div className="bg-white p-6 sm:p-8 rounded-[24px] border border-black/[0.06] shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-black/[0.04]">
            <h2 className="text-base font-semibold text-[#1d1d1f] flex items-center space-x-2">
              <Upload className="w-4 h-4 text-[#1d1d1f]" />
              <span>4. Visual & Acoustic Media Uploads</span>
            </h2>
            <span className="text-[11px] text-[#86868b]">Optional perception inputs</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {/* Image Upload */}
            <div className="border border-dashed border-black/[0.12] bg-[#fbfbfd] hover:bg-white rounded-[20px] p-5 text-center space-y-3 transition-all">
              {imagePreview ? (
                <div className="space-y-3">
                  <img src={imagePreview} alt="Preview" className="h-32 mx-auto rounded-[14px] object-cover shadow-xs" />
                  <div className="flex items-center justify-between text-xs text-[#6e6e73]">
                    <span className="truncate max-w-[150px]">{imageFile?.name}</span>
                    <button
                      type="button"
                      onClick={() => { setImageFile(null); setImagePreview(null); }}
                      className="text-[#ff3b30] hover:underline flex items-center space-x-1"
                    >
                      <Trash2 className="w-3 h-3" />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>
              ) : (
                <label className="cursor-pointer block space-y-2 py-4">
                  <div className="w-10 h-10 rounded-full bg-[#f5f5f7] text-[#1d1d1f] flex items-center justify-center mx-auto">
                    <ImageIcon className="w-5 h-5" />
                  </div>
                  <div className="text-xs font-medium text-[#1d1d1f]">Upload Visual Image</div>
                  <div className="text-[11px] text-[#86868b]">PNG, JPEG, WebP up to 10MB</div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </label>
              )}
            </div>

            {/* Audio Upload */}
            <div className="border border-dashed border-black/[0.12] bg-[#fbfbfd] hover:bg-white rounded-[20px] p-5 text-center space-y-3 transition-all">
              {audioFile ? (
                <div className="space-y-3 py-2">
                  <div className="w-10 h-10 rounded-full bg-[#f5f5f7] text-[#1d1d1f] flex items-center justify-center mx-auto">
                    <Volume2 className="w-5 h-5" />
                  </div>
                  <div className="text-xs text-[#1d1d1f] font-medium">{audioFile.name}</div>
                  <div className="text-[11px] text-[#86868b]">{formatFileSize(audioFile.size)}</div>
                  <button
                    type="button"
                    onClick={() => setAudioFile(null)}
                    className="text-[#ff3b30] text-xs hover:underline flex items-center space-x-1 mx-auto"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Remove</span>
                  </button>
                </div>
              ) : (
                <label className="cursor-pointer block space-y-2 py-4">
                  <div className="w-10 h-10 rounded-full bg-[#f5f5f7] text-[#1d1d1f] flex items-center justify-center mx-auto">
                    <Volume2 className="w-5 h-5" />
                  </div>
                  <div className="text-xs font-medium text-[#1d1d1f]">Upload Acoustic Audio</div>
                  <div className="text-[11px] text-[#86868b]">WAV, MP3 up to 25MB</div>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleAudioChange}
                    className="hidden"
                  />
                </label>
              )}
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-center pt-2">
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full sm:w-auto min-w-[240px] bg-[#1d1d1f] hover:bg-black disabled:bg-[#86868b] text-white font-medium py-3.5 px-8 rounded-full shadow-[0_2px_8px_rgba(0,0,0,0.12)] hover:shadow-[0_4px_16px_rgba(0,0,0,0.18)] transition-all duration-200 flex items-center justify-center space-x-2 text-sm active:scale-98"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Synthesizing Multimodal Diagnosis...</span>
              </>
            ) : (
              <>
                <span>Run Diagnostic Reasoning</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
