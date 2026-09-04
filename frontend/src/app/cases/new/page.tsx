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
  HelpCircle
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
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-industrial-900">New Diagnostic Case</h1>
        <p className="text-sm text-industrial-500">
          Enter equipment specifications, physical sensor telemetry, technician observations, and upload image/audio files for AI multimodal reasoning.
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-start space-x-3 text-rose-800 text-sm">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 text-rose-600 mt-0.5" />
          <div>
            <div className="font-semibold">Submission Failed</div>
            <div>{errorMessage}</div>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Section 1: Equipment Profile */}
        <div className="bg-white p-6 rounded-xl border border-industrial-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-industrial-900 flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-brand-blue" />
            <span>1. Equipment Metadata</span>
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
            <div>
              <label className="block text-xs font-semibold text-industrial-700 mb-1">Equipment Type *</label>
              <select
                value={equipmentType}
                onChange={(e) => setEquipmentType(e.target.value)}
                className="w-full border border-industrial-300 rounded-lg px-3 py-2 bg-white text-industrial-900 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none"
              >
                <option value="motor">Electric Motor</option>
                <option value="pump">Centrifugal Pump</option>
                <option value="gearbox">Industrial Gearbox</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-700 mb-1">Manufacturer</label>
              <input
                type="text"
                value={manufacturer}
                onChange={(e) => setManufacturer(e.target.value)}
                className="w-full border border-industrial-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none"
                placeholder="e.g. Siemens"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-700 mb-1">Model Identifier</label>
              <input
                type="text"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full border border-industrial-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none"
                placeholder="e.g. M-4500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-700 mb-1">Asset Serial Number</label>
              <input
                type="text"
                value={serialNumber}
                onChange={(e) => setSerialNumber(e.target.value)}
                className="w-full border border-industrial-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none"
                placeholder="e.g. SN-94821"
              />
            </div>
          </div>
        </div>

        {/* Section 2: Technician Notes */}
        <div className="bg-white p-6 rounded-xl border border-industrial-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-industrial-900 flex items-center space-x-2">
            <FileText className="w-4 h-4 text-brand-blue" />
            <span>2. Problem Description & Observations</span>
          </h2>
          <div>
            <textarea
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe symptoms, operating sounds, or visual abnormalities..."
              className="w-full border border-industrial-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none text-industrial-900"
            />
          </div>
        </div>

        {/* Section 3: Sensor Readings */}
        <div className="bg-white p-6 rounded-xl border border-industrial-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-industrial-900 flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-brand-blue" />
            <span>3. Physical Sensor Telemetry</span>
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
            <div>
              <label className="block text-xs font-semibold text-industrial-700 mb-1">Temperature (°C)</label>
              <input
                type="number"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="w-full border border-industrial-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none"
                placeholder="e.g. 88.0"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-700 mb-1">Vibration (mm/s RMS)</label>
              <input
                type="number"
                step="0.1"
                value={vibration}
                onChange={(e) => setVibration(e.target.value)}
                className="w-full border border-industrial-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none"
                placeholder="e.g. 7.4"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-700 mb-1">Operating Speed (RPM)</label>
              <input
                type="number"
                value={rpm}
                onChange={(e) => setRpm(e.target.value)}
                className="w-full border border-industrial-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-blue focus:outline-none"
                placeholder="e.g. 1490"
              />
            </div>
          </div>
        </div>

        {/* Section 4: Visual & Audio Media Uploads */}
        <div className="bg-white p-6 rounded-xl border border-industrial-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-industrial-900 flex items-center space-x-2">
            <Upload className="w-4 h-4 text-brand-blue" />
            <span>4. Visual & Acoustic Media Uploads</span>
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {/* Image Upload */}
            <div className="border-2 border-dashed border-industrial-200 rounded-xl p-4 text-center space-y-3">
              {imagePreview ? (
                <div className="space-y-2">
                  <img src={imagePreview} alt="Preview" className="h-32 mx-auto rounded object-cover" />
                  <div className="flex items-center justify-between text-xs text-industrial-600">
                    <span>{imageFile?.name}</span>
                    <button
                      type="button"
                      onClick={() => { setImageFile(null); setImagePreview(null); }}
                      className="text-rose-600 hover:underline flex items-center space-x-1"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>
              ) : (
                <label className="cursor-pointer block space-y-2 py-4">
                  <ImageIcon className="w-8 h-8 text-industrial-400 mx-auto" />
                  <div className="text-xs font-semibold text-brand-blue">Upload Equipment Image</div>
                  <div className="text-[10px] text-industrial-400">PNG, JPG up to 15MB</div>
                  <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
                </label>
              )}
            </div>

            {/* Audio Upload */}
            <div className="border-2 border-dashed border-industrial-200 rounded-xl p-4 text-center space-y-3">
              {audioFile ? (
                <div className="space-y-2 py-4">
                  <Volume2 className="w-8 h-8 text-brand-blue mx-auto" />
                  <div className="text-xs font-semibold text-industrial-800">{audioFile.name}</div>
                  <div className="text-[10px] text-industrial-400">{formatFileSize(audioFile.size)}</div>
                  <button
                    type="button"
                    onClick={() => setAudioFile(null)}
                    className="text-rose-600 text-xs hover:underline flex items-center justify-center space-x-1 mx-auto"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Remove</span>
                  </button>
                </div>
              ) : (
                <label className="cursor-pointer block space-y-2 py-4">
                  <Volume2 className="w-8 h-8 text-industrial-400 mx-auto" />
                  <div className="text-xs font-semibold text-brand-blue">Upload Audio Recording</div>
                  <div className="text-[10px] text-industrial-400">WAV, MP3 up to 25MB</div>
                  <input type="file" accept="audio/*" onChange={handleAudioChange} className="hidden" />
                </label>
              )}
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end pt-4">
          <button
            type="submit"
            disabled={isSubmitting}
            className="bg-brand-blue hover:bg-sky-600 disabled:opacity-50 text-white font-bold px-8 py-3 rounded-xl shadow-md transition flex items-center space-x-2"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Analyzing Multimodal Evidence...</span>
              </>
            ) : (
              <span>Run Diagnostic Analysis</span>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
