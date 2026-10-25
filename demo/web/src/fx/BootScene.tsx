// src/fx/BootScene.tsx
// Boot animation: convergent particles → hexagonal prism wireframe materializes.
// Shown once on first load; fades out after ~3 seconds.

import { useEffect, useRef, useState } from 'react'

interface Props {
  onDone: () => void
}

export default function BootScene({ onDone }: Props) {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const frameRef   = useRef<number>(0)
  const mountRef   = useRef(true)
  const [progress, setProgress] = useState(0)
  const [lines, setLines] = useState<string[]>(['INITIALIZING TRIAD-13 INFERENCE CORE…'])

  useEffect(() => {
    mountRef.current = true
    const LOG_LINES = [
      'INITIALIZING TRIAD-13 INFERENCE CORE…',
      'LOADING MODEL REGISTRY [CONFLUX / QUORUM / NEXUS]',
      'VERIFYING SHA256 ARTIFACT MANIFEST…',
      'PRELOADING 40 CURATED FAULT CASES…',
      'CEN CHILE GRID — TELEMETRY CHANNELS READY',
      'MULTIMODAL FUSION ENGINES ONLINE ✓',
    ]

    let frame = 0
    const TOTAL = 180   // ~3 s at 60 fps
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const W = canvas.width  = 200
    const H = canvas.height = 200
    const cx = W / 2, cy = H / 2

    // Particles
    const N = 60
    const particles = Array.from({ length: N }, (_, i) => ({
      angle: (i / N) * Math.PI * 2,
      radius: 80 + Math.random() * 20,
      speed: 0.004 + Math.random() * 0.003,
      targetR: 30 + Math.random() * 20,
    }))

    // Hex vertices
    const hexVerts = Array.from({ length: 6 }, (_, i) => ({
      x: cx + 50 * Math.cos((i / 6) * Math.PI * 2 - Math.PI / 6),
      y: cy + 50 * Math.sin((i / 6) * Math.PI * 2 - Math.PI / 6),
    }))

    const tick = () => {
      if (!mountRef.current) return
      frameRef.current = requestAnimationFrame(tick)
      frame++

      const t = Math.min(frame / TOTAL, 1)
      setProgress(Math.round(t * 100))

      // Update log lines
      const lineIdx = Math.min(Math.floor(t * LOG_LINES.length), LOG_LINES.length - 1)
      setLines(LOG_LINES.slice(0, lineIdx + 1))

      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = '#05080d'
      ctx.fillRect(0, 0, W, H)

      // Draw hex prism wireframe emerging
      const hexAlpha = Math.max(0, (t - 0.4) / 0.6)
      if (hexAlpha > 0) {
        ctx.strokeStyle = `rgba(56,232,255,${hexAlpha})`
        ctx.lineWidth = 1
        ctx.beginPath()
        hexVerts.forEach((v, i) => {
          if (i === 0) ctx.moveTo(v.x, v.y); else ctx.lineTo(v.x, v.y)
        })
        ctx.closePath()
        ctx.stroke()

        // Top cap (offset)
        ctx.beginPath()
        hexVerts.forEach((v, i) => {
          const tx = v.x + 8 * hexAlpha, ty = v.y - 12 * hexAlpha
          if (i === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty)
        })
        ctx.closePath()
        ctx.stroke()

        // Vertical edges
        hexVerts.forEach(v => {
          ctx.beginPath()
          ctx.moveTo(v.x, v.y)
          ctx.lineTo(v.x + 8 * hexAlpha, v.y - 12 * hexAlpha)
          ctx.stroke()
        })
      }

      // Draw convergent particles
      particles.forEach(p => {
        const eased = t < 0.6 ? 0 : (t - 0.6) / 0.4
        const r = p.radius * (1 - eased) + p.targetR * eased
        p.angle += p.speed
        const x = cx + r * Math.cos(p.angle)
        const y = cy + r * Math.sin(p.angle)
        const alpha = Math.max(0, 1 - eased * 1.5)
        ctx.fillStyle = `rgba(93,255,176,${alpha * 0.8})`
        ctx.beginPath()
        ctx.arc(x, y, 1.5, 0, Math.PI * 2)
        ctx.fill()
      })

      // Central glow
      const glowA = Math.min(t * 2, 1)
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 30)
      grad.addColorStop(0, `rgba(56,232,255,${glowA * 0.3})`)
      grad.addColorStop(1, 'transparent')
      ctx.fillStyle = grad
      ctx.beginPath()
      ctx.arc(cx, cy, 30, 0, Math.PI * 2)
      ctx.fill()

      if (t >= 1) {
        cancelAnimationFrame(frameRef.current)
        setTimeout(onDone, 400)
      }
    }
    tick()

    return () => {
      mountRef.current = false
      cancelAnimationFrame(frameRef.current)
    }
  }, [onDone])

  return (
    <div className="boot-overlay">
      <canvas ref={canvasRef} className="boot-canvas" />
      <div className="boot-text">
        {lines.map((l, i) => (
          <div key={i} style={{ opacity: i === lines.length - 1 ? 1 : 0.4, marginBottom: 2 }}>{l}</div>
        ))}
      </div>
      <div className="boot-progress">
        <div className="boot-progress__fill" style={{ width: `${progress}%` }} />
      </div>
    </div>
  )
}
