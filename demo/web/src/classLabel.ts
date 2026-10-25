// src/classLabel.ts — single source of truth for fault class display names.
const CLASS_LABELS: Record<string, string> = {
  line_fault:        'LINE FAULT',
  breaker_fault:     'BREAKER FAULT',
  transformer_fault: 'TRANSFORMER FAULT',
  busbar_fault:      'BUSBAR FAULT',
}
export function classLabel(id: string): string {
  return CLASS_LABELS[id] ?? id.replace(/_/g, ' ').toUpperCase()
}
