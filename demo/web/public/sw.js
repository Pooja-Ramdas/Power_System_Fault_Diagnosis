// demo/web/public/sw.js
// Service worker: cache-first for /data and /img, network-first for everything else.
// No external URLs are ever fetched (self-hosted fonts, no CDN).

const CACHE_NAME = 'triad13-v1'
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/data/index.json',
]

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  )
  self.skipWaiting()
})

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  )
  self.clients.claim()
})

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url)

  // Only handle same-origin GET requests
  if (event.request.method !== 'GET') return
  if (url.origin !== self.location.origin) return

  const path = url.pathname

  // Cache-first: static data and images
  if (path.startsWith('/data/') || path.startsWith('/img/') || path.startsWith('/fonts/')) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match(event.request)
        if (cached) return cached
        const resp = await fetch(event.request)
        if (resp.ok) cache.put(event.request, resp.clone())
        return resp
      })
    )
    return
  }

  // Network-first for everything else
  event.respondWith(
    fetch(event.request).catch(() =>
      caches.match(event.request).then(r => r ?? new Response('Offline', { status: 503 }))
    )
  )
})
