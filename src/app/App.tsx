/**
 * App root: picks the screen for the current route and owns the app's single camera session.
 *
 * The session lives here rather than in a screen so the camera can be opened directly inside the
 * tap that leaves Home (iOS: start the camera from a user gesture), before the next screen has
 * even rendered. Later screens (scanning pieces) will share the same session.
 */

import { useState } from 'react'
import { createCameraSession } from '../camera/session.ts'
import { CameraCheckScreen } from './camera-check/CameraCheckScreen.tsx'
import { HomeScreen } from './HomeScreen.tsx'
import { useRoute } from './router.ts'

export function App() {
  const { route, navigate, back } = useRoute()
  const [camera] = useState(createCameraSession)

  switch (route) {
    case 'home':
      return (
        <HomeScreen
          onCameraCheck={() => {
            camera.start()
            navigate('camera-check')
          }}
        />
      )
    case 'camera-check':
      return <CameraCheckScreen camera={camera} onBack={back} />
  }
}
