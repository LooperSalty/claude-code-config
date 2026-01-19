# Charger un modèle 3D (GLTF/GLB)

Importe et affiche des modèles 3D au format GLTF/GLB.

## Installation

```bash
npm install @react-three/drei
```

## Génération du composant avec gltfjsx

```bash
npx gltfjsx model.glb --transform --types
```

## Template manuel

```tsx
import { useGLTF, useAnimations } from '@react-three/drei'
import { useEffect, useRef } from 'react'
import { Group } from 'three'

interface ModelProps {
  url: string
  scale?: number
  position?: [number, number, number]
  rotation?: [number, number, number]
}

export function Model({ url, scale = 1, position = [0, 0, 0], rotation = [0, 0, 0] }: ModelProps) {
  const group = useRef<Group>(null)
  const { scene, animations } = useGLTF(url)
  const { actions, names } = useAnimations(animations, group)

  useEffect(() => {
    // Jouer la première animation
    if (names.length > 0 && actions[names[0]]) {
      actions[names[0]]?.reset().fadeIn(0.5).play()
    }
  }, [actions, names])

  return (
    <group ref={group} position={position} rotation={rotation} scale={scale}>
      <primitive object={scene} />
    </group>
  )
}

// Preload du modèle
useGLTF.preload('/models/model.glb')
```

## Avec animations contrôlées

```tsx
import { useGLTF, useAnimations } from '@react-three/drei'
import { useControls, button } from 'leva'
import { useEffect, useRef } from 'react'
import { Group } from 'three'

export function AnimatedModel({ url }: { url: string }) {
  const group = useRef<Group>(null)
  const { scene, animations } = useGLTF(url)
  const { actions, names } = useAnimations(animations, group)

  const { animation } = useControls({
    animation: { options: names },
  })

  useEffect(() => {
    // Arrêter toutes les animations
    Object.values(actions).forEach((action) => action?.stop())

    // Jouer l'animation sélectionnée
    if (animation && actions[animation]) {
      actions[animation]?.reset().fadeIn(0.5).play()
    }
  }, [animation, actions])

  return (
    <group ref={group}>
      <primitive object={scene} />
    </group>
  )
}
```

## Optimisation des modèles

```tsx
import { useGLTF, Instances, Instance } from '@react-three/drei'

// Pour instancier le même modèle plusieurs fois
export function InstancedModels({ count = 100 }: { count: number }) {
  const { nodes, materials } = useGLTF('/models/tree.glb')

  return (
    <Instances
      geometry={nodes.Tree.geometry}
      material={materials.TreeMaterial}
    >
      {Array.from({ length: count }).map((_, i) => (
        <Instance
          key={i}
          position={[Math.random() * 50 - 25, 0, Math.random() * 50 - 25]}
          rotation={[0, Math.random() * Math.PI * 2, 0]}
          scale={0.5 + Math.random() * 0.5}
        />
      ))}
    </Instances>
  )
}
```

## Draco compression

```tsx
import { useGLTF, DRACOLoader } from '@react-three/drei'

// Configurer Draco pour les modèles compressés
useGLTF.setDecoderPath('/draco/')

export function CompressedModel({ url }: { url: string }) {
  const { scene } = useGLTF(url, true) // true = use Draco
  return <primitive object={scene} />
}
```

Quel type de modèle veux-tu charger ?
