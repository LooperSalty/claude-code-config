# Initialiser un projet React Three Fiber

Crée un projet Vite + React + TypeScript + React Three Fiber.

## Commandes à exécuter

```bash
npm create vite@latest $ARGUMENTS -- --template react-ts
cd $ARGUMENTS
npm install three @types/three @react-three/fiber @react-three/drei @react-three/postprocessing leva
```

## Structure à créer

```
src/
├── components/
│   ├── Scene.tsx
│   ├── Experience.tsx
│   └── UI/
│       └── Controls.tsx
├── shaders/
│   └── example/
│       ├── vertex.glsl
│       └── fragment.glsl
├── hooks/
│   └── useAnimationFrame.ts
├── App.tsx
└── main.tsx
```

## Fichiers à générer

### src/App.tsx
```tsx
import { Canvas } from '@react-three/fiber'
import { Experience } from './components/Experience'

export default function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Canvas
        camera={{ position: [0, 2, 5], fov: 75 }}
        gl={{ antialias: true }}
      >
        <Experience />
      </Canvas>
    </div>
  )
}
```

### src/components/Experience.tsx
```tsx
import { OrbitControls, Environment, Grid } from '@react-three/drei'
import { useControls } from 'leva'
import { Scene } from './Scene'

export function Experience() {
  const { gridVisible } = useControls({
    gridVisible: true,
  })

  return (
    <>
      <OrbitControls makeDefault />
      <Environment preset="city" />

      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />

      {gridVisible && (
        <Grid
          args={[20, 20]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#6f6f6f"
          sectionSize={2}
          fadeDistance={30}
        />
      )}

      <Scene />
    </>
  )
}
```

### src/components/Scene.tsx
```tsx
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh } from 'three'

export function Scene() {
  const meshRef = useRef<Mesh>(null)

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.5
    }
  })

  return (
    <mesh ref={meshRef} position={[0, 1, 0]} castShadow>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="hotpink" />
    </mesh>
  )
}
```

### src/hooks/useAnimationFrame.ts
```tsx
import { useRef, useEffect } from 'react'

export function useAnimationFrame(callback: (deltaTime: number) => void) {
  const requestRef = useRef<number>()
  const previousTimeRef = useRef<number>()

  useEffect(() => {
    const animate = (time: number) => {
      if (previousTimeRef.current !== undefined) {
        const deltaTime = time - previousTimeRef.current
        callback(deltaTime)
      }
      previousTimeRef.current = time
      requestRef.current = requestAnimationFrame(animate)
    }

    requestRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(requestRef.current!)
  }, [callback])
}
```

Crée le projet avec cette structure.
