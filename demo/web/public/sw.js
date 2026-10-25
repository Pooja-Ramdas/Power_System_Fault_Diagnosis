// demo/web/public/sw.js
// Service worker: network-first for /data, cache-first for /img and fonts.
const CACHE_NAME = 'triad13-2131a07e'

self.addEventListener('install', event => {
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
  if (event.request.method !== 'GET') return
  if (url.origin !== self.location.origin) return

  const path = url.pathname

  // Network-first for /data (always get fresh cache JSON)
  if (path.startsWith('/data/')) {
    event.respondWith(
      fetch(event.request).then(resp => {
        if (resp.ok) {
          const clone = resp.clone()
          caches.open(CACHE_NAME).then(c => c.put(event.request, clone))
        }
        return resp
      }).catch(() =>
        caches.match(event.request).then(r => r ?? new Response('Offline', { status: 503 }))
      )
    )
    return
  }

  // Cache-first for images and fonts
  if (path.startsWith('/img/') || path.startsWith('/fonts/')) {
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
