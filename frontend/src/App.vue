<template>
  <main class="page-shell">
    <section class="header">
      <div>
        <h1>Codefender 多模型恶意文件检测</h1>
        <p>南开大学反病毒实验室NKAMG - 支持33个检测模型</p>
      </div>
      <el-tag size="large" type="info">{{ backendLabel }}</el-tag>
    </section>

    <section class="workbench">
      <div class="upload-zone" @drop="handleDrop" @dragover.prevent>
        <input ref="fileInput" class="hidden-input" type="file" @change="handleFileChange" />
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-title">上传PE样本进行检测</div>
        <div class="upload-subtitle">后端会先校验PE结构，计算基础信息，再提交给模型分析。</div>
        <el-button type="primary" :loading="loading" @click="selectFile">选择待检PE文件</el-button>
      </div>

      <el-collapse class="models-collapse">
        <el-collapse-item :title="`支持模型清单（${modelNames.length}个）`" name="models">
          <div class="model-chip-grid">
            <el-tag v-for="model in modelNames" :key="model" effect="plain">{{ model }}</el-tag>
          </div>
        </el-collapse-item>
      </el-collapse>
    </section>

    <section v-if="uploadResult" class="results">
      <div class="result-head">
        <div>
          <h2>检测报告如下</h2>
          <div class="tags">
            <el-tag type="success">PE结构：有效</el-tag>
            <el-tag v-if="ensembleResult" :type="ensembleResult.result === '恶意' ? 'danger' : 'success'">
              多模型结果：{{ ensembleResult.result }}
            </el-tag>
            <el-tag v-if="ensembleResult?.virus_name" type="warning">
              病毒名称：{{ ensembleResult.virus_name }}
            </el-tag>
          </div>
        </div>
        <el-radio-group v-model="showSection">
          <el-radio-button v-for="item in sectionOptions" :key="item.value" :label="item.value">
            {{ item.label }}
          </el-radio-button>
        </el-radio-group>
      </div>

      <div v-show="showSection === 'fileInfo'" class="panel">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="文件名">{{ uploadResult.filename || '-' }}</el-descriptions-item>
          <el-descriptions-item label="SHA256">
            <span class="mono break-text">{{ uploadResult.sha256 || '-' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="MD5">
            <span class="mono break-text">{{ uploadResult.md5 || uploadResult.query_result?.MD5 || '-' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="文件大小">{{ formatFileSize(uploadResult.file_size) }}</el-descriptions-item>
          <el-descriptions-item label="文件类型">{{ uploadResult.file_type || '-' }}</el-descriptions-item>
          <el-descriptions-item label="检测时间">{{ uploadResult.detection_time || '-' }}</el-descriptions-item>
        </el-descriptions>

        <h3 class="subhead">PE结构信息</h3>
        <el-descriptions :column="2" border>
          <el-descriptions-item label="PE格式">{{ peInfo.format || '-' }}</el-descriptions-item>
          <el-descriptions-item label="架构">{{ peInfo.architecture || '-' }}</el-descriptions-item>
          <el-descriptions-item label="Machine">{{ peInfo.machine || '-' }}</el-descriptions-item>
          <el-descriptions-item label="子系统">{{ peInfo.subsystem || '-' }}</el-descriptions-item>
          <el-descriptions-item label="入口点">{{ peInfo.entry_point || '-' }}</el-descriptions-item>
          <el-descriptions-item label="镜像基址">{{ peInfo.image_base || '-' }}</el-descriptions-item>
          <el-descriptions-item label="节数量">{{ peInfo.number_of_sections ?? '-' }}</el-descriptions-item>
          <el-descriptions-item label="编译时间">{{ peInfo.compile_time || '-' }}</el-descriptions-item>
          <el-descriptions-item label="SizeOfImage">{{ formatFileSize(peInfo.size_of_image) }}</el-descriptions-item>
          <el-descriptions-item label="SizeOfHeaders">{{ formatFileSize(peInfo.size_of_headers) }}</el-descriptions-item>
        </el-descriptions>

        <h3 class="subhead">节表</h3>
        <el-table :data="peInfo.sections || []" border stripe>
          <el-table-column prop="name" label="名称" min-width="110" />
          <el-table-column prop="virtual_address" label="VirtualAddress" min-width="140" />
          <el-table-column prop="virtual_size" label="VirtualSize" min-width="120" />
          <el-table-column prop="raw_size" label="RawSize" min-width="120" />
          <el-table-column prop="raw_pointer" label="RawPointer" min-width="130" />
          <el-table-column prop="characteristics" label="Characteristics" min-width="150" />
        </el-table>
      </div>

      <div v-show="showSection === 'modelDetection'" class="panel">
        <div class="summary-row">
          <el-tag size="large" type="info">共 {{ modelCount }} 个检测模型</el-tag>
          <el-tag size="large" type="danger">恶意: {{ maliciousCount }}</el-tag>
          <el-tag size="large" type="success">安全: {{ safeCount }}</el-tag>
        </div>
        <el-table :data="modelRows" border stripe class="model-table">
          <el-table-column type="index" label="序号" width="70" />
          <el-table-column prop="model" label="检测模型" min-width="180" />
          <el-table-column label="恶意概率" min-width="180">
            <template #default="{ row }">
              <el-progress :percentage="getProbabilityPercent(row.probability)" :color="getProbabilityColor(row.probability)" />
            </template>
          </el-table-column>
          <el-table-column label="结果" width="120">
            <template #default="{ row }">
              <el-tag :type="row.result === '恶意' ? 'danger' : row.result === '安全' ? 'success' : 'info'">
                {{ row.result }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="virus_name" label="病毒名称" min-width="180">
            <template #default="{ row }">{{ row.virus_name || '-' }}</template>
          </el-table-column>
        </el-table>
      </div>
    </section>
  </main>
</template>

<script setup>
import axios from 'axios'
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'

const apiService = axios.create({ timeout: 600000 })

const showSection = ref('fileInfo')
const loading = ref(false)
const uploadResult = ref(null)
const apiBaseUrl = ref('http://127.0.0.1:5005')
const fileInput = ref(null)
const modelNames = ref([
  'MalConv', 'MalConv2', 'ByteTransformer', 'SaxeBerlinDNN', 'EmberDNN',
  'NatarajKNN', 'DrebinSVM', 'DL4MD', 'PEMinerRF', 'DrebinImport',
  'ALOHANet', 'EmberGBDT_Import', 'RaffFeatureNet', 'DrebinString',
  'OpcodeLSTM', 'OpcodeTransformer', 'OpcodeStatNet', 'AsmEmbedNet',
  'MalGraph', 'CFGGAT', 'CFGGCN', 'CFGDGCNN', 'CallGraphGNN',
  'GraphStatNet', 'GrayscaleCNN', 'InceptionV3', 'IMCFN', 'ColorCNN',
  'MarkovCNN', 'EntropyMapCNN', 'HashEmbedNet', 'MetadataNet', 'ResourceNet'
])

const sectionOptions = [
  { label: '基础信息', value: 'fileInfo' },
  { label: '模型检测', value: 'modelDetection' }
]

const backendLabel = computed(() => apiBaseUrl.value.replace(/^https?:\/\//, ''))
const peInfo = computed(() => uploadResult.value?.pe_info || {})

const modelResults = computed(() => {
  const results = uploadResult.value?.exe_result || uploadResult.value?.results || {}
  return Object.fromEntries(Object.entries(results).filter(([model]) => model !== '集成结果' && model !== 'error'))
})

const ensembleResult = computed(() => {
  const results = uploadResult.value?.exe_result || uploadResult.value?.results || {}
  return results['集成结果'] || null
})

const modelRows = computed(() => Object.entries(modelResults.value).map(([model, data]) => ({
  model,
  probability: data.probability,
  result: data.result,
  virus_name: data.virus_name
})))

const modelCount = computed(() => modelRows.value.length)
const maliciousCount = computed(() => modelRows.value.filter(row => row.result === '恶意').length)
const safeCount = computed(() => modelRows.value.filter(row => row.result === '安全').length)

onMounted(async () => {
  await loadConfig()
  await loadModels()
})

async function loadConfig() {
  try {
    const response = await apiService.get('/config.ini', { responseType: 'text', timeout: 5000 })
    const parsed = parseIni(response.data)
    apiBaseUrl.value = parsed.codefender?.baseUrl || parsed.api?.baseUrl || apiBaseUrl.value
  } catch (error) {
    console.warn('加载配置文件失败:', error.message)
  }
}

async function loadModels() {
  try {
    const response = await apiService.get(`${apiBaseUrl.value}/models`, { timeout: 5000 })
    if (Array.isArray(response.data?.models)) {
      modelNames.value = response.data.models
    }
  } catch (error) {
    console.warn('加载模型清单失败:', error.message)
  }
}

function parseIni(text) {
  const result = {}
  let section = ''
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith(';')) continue
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      section = trimmed.slice(1, -1)
      result[section] = result[section] || {}
      continue
    }
    const index = trimmed.indexOf('=')
    if (index > -1 && section) {
      result[section][trimmed.slice(0, index).trim()] = trimmed.slice(index + 1).trim()
    }
  }
  return result
}

function resetResult() {
  uploadResult.value = null
  showSection.value = 'fileInfo'
}

function selectFile() {
  if (!loading.value) fileInput.value?.click()
}

function handleFileChange(event) {
  const file = event.target.files?.[0]
  if (file) uploadFile(file)
  event.target.value = ''
}

function handleDrop(event) {
  event.preventDefault()
  if (loading.value) return
  const files = event.dataTransfer.files
  if (files.length !== 1) {
    ElMessage.error('只支持上传一个文件')
    return
  }
  uploadFile(files[0])
}

async function uploadFile(file) {
  resetResult()
  loading.value = true
  const formData = new FormData()
  formData.append('file', file)
  try {
    const response = await apiService.post(`${apiBaseUrl.value}/detect`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    uploadResult.value = normalizeDetectionResponse(response.data)
    ElMessage.success('检测完成')
  } catch (error) {
    ElMessage.error('检测失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loading.value = false
  }
}

function normalizeDetectionResponse(data) {
  const queryResult = data.query_result || {}
  return {
    ...data,
    filename: data.filename || data.original_filename || queryResult.name || '-',
    sha256: data.sha256 || queryResult['SHA-256'] || queryResult.SHA256 || '',
    md5: data.md5 || queryResult.MD5 || '',
    file_size: data.file_size || queryResult['文件大小'] || 0,
    file_type: data.file_type || queryResult['文件类型'] || queryResult['类型'] || '-',
    detection_time: data.detection_time || new Date().toLocaleString(),
    exe_result: data.exe_result || data.results || {},
    pe_info: data.pe_info || {}
  }
}

function formatFileSize(value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'string' && value.includes('bytes')) {
    value = Number.parseInt(value, 10)
  } else if (typeof value === 'string') {
    return value
  }
  const bytes = Number(value)
  if (!Number.isFinite(bytes)) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes || 1) / Math.log(1024)), units.length - 1)
  return `${Math.round((bytes / Math.pow(1024, index)) * 100) / 100} ${units[index]}`
}

function getProbabilityPercent(probability) {
  if (probability === null || probability === undefined) return 0
  if (typeof probability === 'string') return Number.parseFloat(probability.replace('%', '')) || 0
  return Math.round(Number(probability) * 10000) / 100
}

function getProbabilityColor(probability) {
  const percent = getProbabilityPercent(probability)
  if (percent >= 70) return '#d94f45'
  if (percent >= 40) return '#b7791f'
  return '#2f855a'
}
</script>
