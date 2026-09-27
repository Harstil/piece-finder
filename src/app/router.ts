/**
 * A tiny state-based router: the app has a handful of screens and no shareable deep links, so a
 * router library would be all cost.
 *
 * Each navigation pushes a history entry (the URL stays the same), so Android's back button and
 * the browser's back gesture return to the previous screen instead of leaving the app. A reload
 * always starts on Home: screens like the camera must be entered by a tap (see
 * src/camera/session.ts), and the entry the reload landed on is relabelled as Home to match.
 */

import { useCallback, useEffect, useState } from 'react'

export type Route = 'home' | 'setup' | 'scan' | 'picture' | 'camera-check'

const ROUTES: readonly Route[] = ['home', 'setup', 'scan', 'picture', 'camera-check']

function routeFromHistoryState(state: unknown): Route {
  const route = (state as { route?: unknown } | null)?.route
  return ROUTES.find((known) => known === route) ?? 'home'
}

export function useRoute(): { route: Route; navigate: (to: Route) => void; replace: (to: Route) => void; back: () => void } {
  const [route, setRoute] = useState<Route>('home')

  useEffect(() => {
    history.replaceState({ route: 'home' }, '')
    const onPopState = (event: PopStateEvent) => setRoute(routeFromHistoryState(event.state))
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  const navigate = useCallback((to: Route) => {
    history.pushState({ route: to }, '')
    setRoute(to)
  }, [])

  const replace = useCallback((to: Route) => {
    history.replaceState({ route: to }, '')
    setRoute(to)
  }, [])

  const back = useCallback(() => history.back(), [])

  return { route, navigate, replace, back }
}
