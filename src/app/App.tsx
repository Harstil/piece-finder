/**
 * App root: picks the screen for the current route and owns the app's single camera session and the
 * active puzzle.
 *
 * The camera session lives here rather than in a screen so it can be opened directly inside the tap
 * that leaves Home or Setup (iOS: start the camera from a user gesture), before the next screen has
 * even rendered. The scanner and the Camera check share it.
 */

import { useState } from 'react'
import { createCameraSession } from '../camera/session.ts'
import { CameraCheckScreen } from './camera-check/CameraCheckScreen.tsx'
import { HomeScreen } from './HomeScreen.tsx'
import { PictureScreen } from './picture/PictureScreen.tsx'
import { useRoute } from './router.ts'
import { ScanScreen } from './scan/ScanScreen.tsx'
import { SetupScreen } from './setup/SetupScreen.tsx'
import { usePuzzle } from './usePuzzle.ts'

export function App() {
  const { route, navigate, replace, back } = useRoute()
  const [camera] = useState(createCameraSession)
  const puzzle = usePuzzle()

  switch (route) {
    case 'home':
      return (
        <HomeScreen
          status={puzzle.status}
          onScan={() => {
            camera.start()
            navigate('scan')
          }}
          onNewPuzzle={() => navigate('setup')}
          onCameraCheck={() => {
            camera.start()
            navigate('camera-check')
          }}
        />
      )
    case 'setup':
      return (
        <SetupScreen
          onCancel={back}
          onStart={(photo, corners, grid) => {
            camera.start()
            replace('scan')
            void puzzle.create(photo, corners, grid)
          }}
        />
      )
    case 'scan':
      return <ScanScreen camera={camera} puzzle={puzzle} onBack={back} onPicture={() => navigate('picture')} />
    case 'picture':
      // Back to the scanner reopens the camera inside the tap (the scanner released it on the way here).
      return puzzle.status.kind === 'ready' ? (
        <PictureScreen
          view={puzzle.status.view}
          puzzle={puzzle}
          onBack={() => {
            camera.start()
            back()
          }}
        />
      ) : (
        <ScanScreen camera={camera} puzzle={puzzle} onBack={back} onPicture={() => {}} />
      )
    case 'camera-check':
      return <CameraCheckScreen camera={camera} onBack={back} />
  }
}
