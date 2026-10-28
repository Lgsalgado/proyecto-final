# Directorio de Datos Originales (data/raw/)

Este directorio contiene las secuencias cinemáticas crudas y videos fuente de captura biomecánica:

- **Formato crudo:** Registros continuos de video RGB 720p/1080p a 30 FPS en planos sagital y anterolateral.
- **Partición por sujeto y video:** 1,200 observaciones segmentadas preservando independencia entre los conjuntos de Entrenamiento (70%), Validación (15%) y Test Set Ciego (15%).
- **Tratamiento:** Los videos crudos son procesados mediante el pipeline de visión (`src/vision/pose_detector.py`) para extraer 33 landmarks tridimensionales, filtrados con Savitzky-Golay e indexados en `data/processed/squat_features.csv`.
- **Privacidad y GDPR/LOPDP:** Los videos con rostros de participantes de validación se anonimizan y procesan localmente sin almacenamiento permanente de biométricos faciales.
