# Consideraciones Éticas y Responsabilidad en IA

**Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza**  
**Maestría en IA (MIAR0545) — Universidad de Especialidades Espíritu Santo (UEES)**  
**Workshop: Impacto Social y Responsabilidad en Tu Proyecto de IA — Semana 4**

---

## 1. Contexto del Proyecto

El sistema detecta y clasifica posturas en ejercicios de fuerza (especialmente sentadillas) mediante visión por computador. Utiliza MediaPipe BlazePose para extraer 33 landmarks 3D en tiempo real, un motor cinemático vectorial basado en estándares NSCA, y una red neuronal `RegularizedSmallMLP` (12,580 parámetros) que clasifica 4 categorías: Técnica Correcta, Valgo Dinámico de Rodilla, Cifosis Lumbar/Butt Wink, y Profundidad Insuficiente.

**Procesamiento Edge AI:** Todo el procesamiento ocurre localmente en el dispositivo del usuario. No se almacena ni transmite video a servidores externos.

**Alcance de implementación:**
- Gimnasios comerciales y centros deportivos
- Entrenamiento en hogar (webcam estándar)
- Estimación de usuarios potenciales: 50,000–200,000 en el primer año
- El sistema **asiste** decisiones (feedback biomecánico) pero **no sustituye** diagnóstico clínico profesional

---

## 2. Análisis de Stakeholders

### 2.1 Tabla de Stakeholders

| Stakeholder | Descripción | Beneficios | Riesgos | Nivel de Poder | Consideración en Diseño |
|:---|:---|:---|:---|:---:|:---|
| **Practicantes de fitness / Atletas amateur** | Usuarios finales que realizan ejercicios de fuerza sin supervisión profesional constante | Retroalimentación biomecánica experta en tiempo real; prevención de lesiones LCA, meniscales y lumbares | Confianza excesiva en el sistema; falsos negativos en posiciones ocluidas | Alto | Interfaz en lenguaje natural; disclaimers claros; umbral de confianza mínimo |
| **Entrenadores personales y preparadores físicos** | Profesionales que supervisan la técnica de sus clientes | Herramienta objetiva de análisis; datos cuantitativos; escalabilidad de supervisión | Percepción de desplazamiento profesional; erosión de autoridad | Alto | Sistema complementario explícito; APIs para integración con sus plataformas |
| **Fisioterapeutas y médicos del deporte** | Profesionales de salud que tratan lesiones musculo-esqueléticas | Datos preventivos cuantitativos; reducción de casos por técnica deficiente | Responsabilidad clínica si el sistema falla; uso fuera de su alcance | Medio-Alto | Métricas auditables NSCA; documentación de limitaciones clínicas explícita |
| **Propietarios de gimnasios y centros deportivos** | Tomadores de decisión institucionales | Diferenciación competitiva; reducción de lesiones en instalaciones; valor añadido | Costo de adopción; resistencia de personal; liability legal | Medio | Edge AI sin infraestructura adicional; privacidad garantizada para sus clientes |

---

## 3. Evaluación de Impacto Social

### 3.1 Impactos Positivos

1. **Democratización de la supervisión biomecánica experta:** El sistema elimina la barrera económica del análisis biomecánico de laboratorio (USD 200–500 por sesión) y lo hace accesible a cualquier usuario con una webcam estándar, democratizando el acceso a retroalimentación experta de calidad.

2. **Prevención primaria de lesiones articulares degenerativas:** Al detectar en tiempo real el valgo dinámico de rodilla (riesgo de LCA), la cifosis lumbar (riesgo discal L4-S1) y la profundidad insuficiente (activación muscular subóptima), el sistema actúa como capa preventiva antes de que ocurra la lesión, reduciendo el riesgo de condromalacia, osteoartritis prematura y hernias discales.

3. **Reducción de costos en rehabilitación:** La prevención primaria de lesiones musculo-esqueléticas puede reducir significativamente los costos de rehabilitación (USD 5,000–30,000 por episodio de LCA) y el ausentismo laboral/deportivo asociado.

4. **Extensión de la carrera deportiva:** Al mantener la integridad biomecánica durante el entrenamiento, el sistema contribuye a prolongar la vida deportiva activa de los practicantes, con beneficios en salud cardiovascular, densidad ósea y bienestar mental.

### 3.2 Impactos Negativos y Riesgos

1. **Riesgo de falsos negativos con consecuencias físicas:** Si el sistema clasifica incorrectamente una técnica deficiente como correcta, el usuario podría continuar con una ejecución lesiva con falsa seguridad. Mitigación: umbral de confianza mínimo de 0.70; disclaimers obligatorios.

2. **Sesgo por variabilidad antropométrica:** El dataset de entrenamiento puede no representar adecuadamente la diversidad de morfologías corporales (longitud de fémur, ratio cadera/hombro en diferentes etnias, personas con limitaciones físicas). Mitigación: normalización antropométrica relativa; validación cruzada estratificada.

3. **Erosión de la autoridad profesional:** La adopción masiva sin contexto puede llevar a usuarios a sustituir la supervisión de entrenadores certificados por el sistema, incrementando el riesgo en ejercicios avanzados fuera del alcance del modelo.

---

## 4. Análisis de Sesgos y Equidad

### 4.1 Tipos de Sesgo Identificados

| Tipo de Sesgo | Descripción | Mitigación Implementada |
|:---|:---|:---|
| **Sesgo de representación** | Dataset sintético puede no reflejar diversidad morfológica real (género, edad, etnia, condición física) | Normalización antropométrica relativa (d_caderas); features adimensionales |
| **Sesgo de medición** | Variabilidad en condiciones de iluminación, ángulo de cámara y distancia afectan la calidad de los landmarks | Umbral de confianza c_i ≥ 0.60; imputación cinemática por simetría bilateral |
| **Sesgo de automatización** | El sistema puede reforzar patrones biomecánicos "promedio" sin considerar variaciones anatómicas individuales válidas | Disclaimer explícito: el sistema provee orientación general, no prescripción individual |
| **Sesgo de confirmación** | Los entrenadores pueden aceptar o rechazar las alertas del sistema según sus creencias previas | Métricas cuantitativas auditables presentadas junto a la clasificación |

### 4.2 Preguntas de Equidad

1. ¿El sistema funciona igualmente bien para usuarios con diferentes morfologías corporales? — Parcialmente: la normalización antropométrica mitiga pero no elimina el sesgo.
2. ¿El sistema es accesible para personas con discapacidades físicas que modifiquen la biomecánica estándar? — No en su versión actual; requiere extensión del modelo.
3. ¿Los errores del sistema afectan desproporcionadamente a algún grupo? — Riesgo identificado en usuarios con proporciones corporales extremas (hiperlordosis, escoliosis).

---

## 5. Privacidad y Protección de Datos

### 5.1 Diseño Privacy-by-Design

| Principio | Implementación |
|:---|:---|
| **Minimización de datos** | Solo se procesan coordenadas 3D de landmarks; nunca se retienen frames de video |
| **Procesamiento local (Edge AI)** | Toda la inferencia ocurre en el dispositivo del usuario; cero transmisión de video |
| **Ausencia de identificación biométrica** | Los landmarks se descartan inmediatamente; no se almacena identidad facial ni corporal |
| **Consentimiento explícito** | El usuario activa la cámara manualmente; puede detener en cualquier momento |

### 5.2 Compliance Regulatorio

| Regulación | Jurisdicción | Cumplimiento |
|:---|:---|:---|
| **LOPDP** (Ley Orgánica de Protección de Datos Personales, R.O. Sup. 459, 2021) | Ecuador | Cumple: procesamiento edge, no retención de video, minimización de datos conforme al Art. 6 |
| **GDPR** (General Data Protection Regulation, 2016/679) | Union Europea | Cumple: base legal legítima (consentimiento explícito), derecho al olvido garantizado, privacy-by-design |
| **Ley Federal de Protección de Datos Personales** (LFPDPPP) | México | Cumple: principios de licitud, consentimiento y minimización |
| **Ley 1581 de 2012** | Colombia | Cumple: autorización expresa del titular, finalidad definida |

---

## 6. Accountability y Gobernanza

### 6.1 Cadena de Responsabilidad

1. **Desarrolladores** (Luis & Daniel Salgado): Responsables del diseño, entrenamiento y documentación del sistema. Obligación de reportar limitaciones técnicas conocidas.
2. **Entrenadores y profesionales de salud**: Responsables de interpretar las alertas del sistema en el contexto clínico de cada usuario.
3. **Usuario final**: Responsable del uso dentro de los límites documentados; el sistema no sustituye evaluación médica.
4. **Institución (UEES)**: Supervisión académica y validación ética del proyecto de investigación.

### 6.2 Plan de Monitoreo Continuo

- Registro de métricas de confianza por sesión (sin datos de usuario).
- Versión semántica del modelo documentada en `models/README.md`.
- Pipeline de re-entrenamiento si el accuracy en producción cae > 5% respecto a la línea base.
- Proceso de reporte de errores para usuarios y profesionales.

---

## 7. Transparencia y Explicabilidad

El sistema provee explicabilidad en dos niveles:

1. **Nivel biomecánico (Motor NSCA):** Cada alerta incluye el ángulo articular cuantitativo que disparó el diagnóstico (ej. "Ángulo de rodilla: 78.4°, profundidad insuficiente según estándar NSCA ≤ 90°").
2. **Nivel del modelo (RegularizedSmallMLP):** Las probabilidades softmax por clase (confianza 0–100%) son accesibles en la respuesta JSON del endpoint `/predict`, permitiendo auditoría de la decisión clasificatoria.

---

## 7. Prevención de Uso Dual y Mal Uso

### ¿Podría el modelo usarse con fines maliciosos?
El sistema está diseñado estrictamente para el análisis cinemático de sentadillas. Sin embargo, se identifican posibles escenarios de mal uso:
1. **Vigilancia encubierta o perfilamiento corporal:** Intentar usar la estimación de pose para rastreo no autorizado de personas.
2. **Evaluación punitiva en entornos laborales o de seguros:** Usar la clasificación automática para denegar cobertura médica o penalizar a trabajadores sin supervisión humana.
3. **Sobreentrenamiento lesivo (gamificación nociva):** Usuarios que fuercen rangos articulares extremos para "engañar" al modelo y obtener 100% de técnica correcta.

### Salvaguardas implementadas:
- **Inferencia estrictamente local en CPU (Edge AI):** Sin capacidades de transmisión remota, telemetría oculta ni almacenamiento de secuencias de video.
- **Desacoplamiento biométrico:** El modelo solo procesa 12 ángulos articulares adimensionales y en grados sexagesimales; no procesa rostros, iris ni identificadores personales.
- **Licencia y Términos de Uso:** Prohibición explícita de integración en sistemas de vigilancia, diagnóstico clínico automatizado sin médico o evaluación laboral punitiva.

---

## 8. Limitaciones Reconocidas y Casos Límite

### ¿En qué casos NO debe usarse el modelo?
- **Población con patologías ortopédicas severas preexistentes:** Prótesis de cadera/rodilla, secuelas de fracturas o asimetrías anatómicas congénitas.
- **Entrenamiento con cargas máximas (1RM):** En levantamientos máximos donde la deformación técnica es momentánea y la cámara puede tener oclusión por múltiples asistentes (*spotters*).
- **Ejercicios multiarticulares diferentes a sentadillas:** No es aplicable para peso muerto, cargadas olímpicas ni arrancadas.

### Advertencias obligatorias a los usuarios:
- El sistema es una **herramienta educativa y de asistencia biomecánica**, no un dispositivo médico validado por la FDA/EMA ni el ARCSA.
- Ninguna métrica del sistema sustituye la evaluación presencial de un fisioterapeuta colegiado o entrenador certificado.

### Casos límite (Edge Cases) donde el modelo pierde confiabilidad:
- **Iluminación deficiente (< 50 lux):** Pérdida de visibilidad de los puntos clave anatómicos.
- **Ropa excesivamente holgada:** Ropa que oculte la posición exacta de las crestas ilíacas o articulación de la rodilla.
- **Perspectiva oblicua extrema (> 30° fuera del plano sagital):** Distorsiona la trigonometría 3D de los ángulos articulares.

---

## 9. Conclusiones Éticas

1. El sistema opera bajo un paradigma de **asistencia, no sustitución** del juicio profesional biomecánico.
2. La arquitectura **Edge AI** garantiza privacidad absoluta sin comprometer la utilidad del sistema.
3. Los sesgos identificados son **mitigables** mediante normalización antropométrica y validación cruzada diversificada.
4. El cumplimiento con **LOPDP Ecuador**, GDPR y regulaciones latinoamericanas está garantizado por diseño arquitectónico.
5. La **explicabilidad cuantitativa** (ángulos NSCA + probabilidades softmax) satisface los requisitos de transparencia algorítmica para aplicaciones de salud y bienestar.

