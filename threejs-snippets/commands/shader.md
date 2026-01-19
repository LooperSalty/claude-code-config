# Créer un Shader GLSL

Génère un shader personnalisé pour Three.js ou React Three Fiber.

## Templates disponibles

### 1. Shader de base
```glsl
// vertex.glsl
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vPosition;

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
```

```glsl
// fragment.glsl
uniform float uTime;
uniform vec3 uColor;

varying vec2 vUv;
varying vec3 vNormal;

void main() {
    vec3 color = uColor;

    // Exemple: gradient basé sur UV
    color = mix(color, vec3(1.0), vUv.y * 0.5);

    gl_FragColor = vec4(color, 1.0);
}
```

### 2. Shader animé (waves)
```glsl
// vertex.glsl
uniform float uTime;
uniform float uAmplitude;
uniform float uFrequency;

varying vec2 vUv;
varying float vElevation;

void main() {
    vUv = uv;

    vec3 pos = position;
    float elevation = sin(pos.x * uFrequency + uTime) * uAmplitude;
    elevation += sin(pos.z * uFrequency * 0.5 + uTime * 0.8) * uAmplitude * 0.5;
    pos.y += elevation;

    vElevation = elevation;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
```

```glsl
// fragment.glsl
uniform vec3 uColorA;
uniform vec3 uColorB;

varying vec2 vUv;
varying float vElevation;

void main() {
    float mixStrength = (vElevation + 0.5) * 0.5;
    vec3 color = mix(uColorA, uColorB, mixStrength);

    gl_FragColor = vec4(color, 1.0);
}
```

### 3. Post-processing (blur)
```glsl
// fragment.glsl
uniform sampler2D tDiffuse;
uniform vec2 uResolution;
uniform float uBlurAmount;

varying vec2 vUv;

void main() {
    vec4 color = vec4(0.0);
    float total = 0.0;

    for (float x = -4.0; x <= 4.0; x++) {
        for (float y = -4.0; y <= 4.0; y++) {
            vec2 offset = vec2(x, y) * uBlurAmount / uResolution;
            color += texture2D(tDiffuse, vUv + offset);
            total += 1.0;
        }
    }

    gl_FragColor = color / total;
}
```

## Utilisation en React Three Fiber

```tsx
import { shaderMaterial } from '@react-three/drei'
import { extend, useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'

import vertexShader from './shaders/vertex.glsl'
import fragmentShader from './shaders/fragment.glsl'

const CustomMaterial = shaderMaterial(
  {
    uTime: 0,
    uColor: new THREE.Color('#ff6b6b'),
  },
  vertexShader,
  fragmentShader
)

extend({ CustomMaterial })

export function CustomMesh() {
  const materialRef = useRef<THREE.ShaderMaterial>(null)

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime
    }
  })

  return (
    <mesh>
      <planeGeometry args={[4, 4, 32, 32]} />
      <customMaterial ref={materialRef} />
    </mesh>
  )
}
```

Quel type de shader veux-tu créer ?
