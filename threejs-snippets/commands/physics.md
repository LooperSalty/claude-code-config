# Ajouter la physique (Rapier/Cannon)

Configure la physique 3D avec @react-three/rapier ou cannon-es.

## Installation

```bash
# Option 1: Rapier (recommandé - plus performant)
npm install @react-three/rapier

# Option 2: Cannon-es
npm install @react-three/cannon
```

## Template Rapier

```tsx
import { Canvas } from '@react-three/fiber'
import { Physics, RigidBody, CuboidCollider } from '@react-three/rapier'

export default function App() {
  return (
    <Canvas>
      <Physics gravity={[0, -9.81, 0]} debug>
        {/* Sol */}
        <RigidBody type="fixed">
          <mesh position={[0, -1, 0]} receiveShadow>
            <boxGeometry args={[20, 0.5, 20]} />
            <meshStandardMaterial color="lightgray" />
          </mesh>
        </RigidBody>

        {/* Cube dynamique */}
        <RigidBody position={[0, 5, 0]} restitution={0.5}>
          <mesh castShadow>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial color="hotpink" />
          </mesh>
        </RigidBody>

        {/* Sphère dynamique */}
        <RigidBody position={[2, 8, 0]} colliders="ball">
          <mesh castShadow>
            <sphereGeometry args={[0.5, 32, 32]} />
            <meshStandardMaterial color="skyblue" />
          </mesh>
        </RigidBody>
      </Physics>
    </Canvas>
  )
}
```

## Template Cannon-es

```tsx
import { Canvas } from '@react-three/fiber'
import { Physics, usePlane, useBox, useSphere } from '@react-three/cannon'

function Ground() {
  const [ref] = usePlane(() => ({
    rotation: [-Math.PI / 2, 0, 0],
    position: [0, -1, 0],
  }))

  return (
    <mesh ref={ref} receiveShadow>
      <planeGeometry args={[20, 20]} />
      <meshStandardMaterial color="lightgray" />
    </mesh>
  )
}

function Cube({ position }: { position: [number, number, number] }) {
  const [ref] = useBox(() => ({
    mass: 1,
    position,
    args: [1, 1, 1],
  }))

  return (
    <mesh ref={ref} castShadow>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="hotpink" />
    </mesh>
  )
}

function Ball({ position }: { position: [number, number, number] }) {
  const [ref] = useSphere(() => ({
    mass: 1,
    position,
    args: [0.5],
  }))

  return (
    <mesh ref={ref} castShadow>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial color="skyblue" />
    </mesh>
  )
}

export default function App() {
  return (
    <Canvas shadows>
      <Physics gravity={[0, -9.81, 0]}>
        <Ground />
        <Cube position={[0, 5, 0]} />
        <Ball position={[2, 8, 0]} />
      </Physics>
    </Canvas>
  )
}
```

## Colliders personnalisés

```tsx
// Collider convex pour modèles 3D
<RigidBody colliders="hull">
  <Model />
</RigidBody>

// Collider trimesh (précis mais plus lent)
<RigidBody colliders="trimesh">
  <Model />
</RigidBody>

// Collider manuel
<RigidBody colliders={false}>
  <CuboidCollider args={[0.5, 1, 0.5]} position={[0, 1, 0]} />
  <Model />
</RigidBody>
```

Quel système de physique veux-tu utiliser ?
