# Informe de Optimización de Parámetros (Subfase 6.5)

Fecha: 2024-08-01

## 1. Resumen Ejecutivo

El objetivo de esta subfase fue ajustar los parámetros del nuevo sistema de gestión de recursos basándose en los resultados de las pruebas de carga y benchmarking. Se realizaron ajustes iterativos principalmente en los umbrales de memoria y el intervalo de monitoreo. La configuración final busca un equilibrio entre la reactividad del sistema a los picos de recursos y la prevención de limpiezas innecesariamente frecuentes, manteniendo la configuración de concurrencia en 'auto' que demostró ser adecuada.

## 2. Parámetros Clave Ajustados

Los siguientes parámetros en `config.yaml` sección `resource_management` fueron modificados:

*   `monitoring_interval`: **30** (Anterior: 60)
*   `warning_threshold_memory`: **75** (Anterior: 70)
*   `aggressive_threshold_memory`: 85 (Anterior: 85) - *Sin cambios*
*   `warning_threshold_cpu`: 80 (Anterior: 80) - *Sin cambios*
*   `concurrency.default_cpu_workers`: "auto" (Anterior: "auto") - *Sin cambios*
*   `concurrency.default_io_workers`: "auto" (Anterior: "auto") - *Sin cambios*
*   `concurrency.max_total_workers`: null (Anterior: null) - *Sin cambios*

## 3. Proceso de Experimentación y Resultados

Se realizaron varios ciclos de ajuste y re-ejecución de benchmarks y pruebas de carga seleccionadas:

*   **Ajuste 1: Reducción de `monitoring_interval` a 30s.**
    *   **Prueba:** Se re-ejecutó `run_concurrent_queries.py` bajo carga moderada y se observó `concurrent_queries_metrics.csv`.
    *   **Observación:** Se detectó una respuesta más rápida del `ResourceManager` a los picos de memoria generados durante la carga de modelos/embeddings para las consultas iniciales. La limpieza agresiva (si se activaba) ocurría antes, previniendo que el uso de memoria alcanzara niveles críticos observados con el intervalo de 60s. No se notó un impacto negativo significativo en el QPS promedio.

*   **Ajuste 2: Aumento de `warning_threshold_memory` a 75s.**
    *   **Prueba:** Se re-ejecutó `run_concurrent_queries.py` (carga ligera, múltiples ejecuciones) y se analizaron los logs del sistema (con nivel INFO temporalmente) para observar la frecuencia de limpieza no agresiva.
    *   **Observación:** Con el umbral anterior de 70%, se observaron limpiezas no agresivas (incluyendo `release_inactive_models`) que ocurrían incluso cuando el sistema no estaba bajo estrés real, potencialmente eliminando modelos/sesiones que podrían reutilizarse pronto. Al subirlo a 75%, la frecuencia de estas limpiezas disminuyó en escenarios de carga baja/media, lo que podría mejorar la latencia "warm-start" de las consultas.

*   **Otros Ajustes (Exploratorios, sin cambio final):** Se probaron brevemente valores más altos para `aggressive_threshold_memory` (e.g., 90%), pero se consideró que 85% ofrecía un margen de seguridad adecuado. La configuración `auto` para los workers de concurrencia pareció funcionar bien en las pruebas de ingesta, adaptándose razonablemente a los cores disponibles, por lo que no se modificó.

## 4. Análisis de Trade-offs

*   **Intervalo de Monitoreo:** Reducir `monitoring_interval` a 30s mejora la reactividad ante picos de recursos, pero introduce un ligero overhead adicional por las comprobaciones más frecuentes. Se consideró un buen compromiso.
*   **Umbral de Advertencia de Memoria:** Aumentar `warning_threshold_memory` a 75% reduce la frecuencia de limpiezas "preventivas", lo que puede mejorar el rendimiento promedio al mantener más recursos cacheados (modelos/sesiones), pero podría permitir que el uso de memoria se acerque más al umbral agresivo antes de actuar. Se priorizó evitar limpiezas innecesarias en carga normal.

## 5. Justificación de la Configuración Final

*   `monitoring_interval: 30`: Ofrece una respuesta más ágil a cambios rápidos en el uso de recursos, crucial durante picos de ingesta o consultas concurrentes, sin parecer introducir un overhead excesivo.
*   `warning_threshold_memory: 75`: Evita la limpieza prematura de recursos que aún podrían ser útiles, optimizando para escenarios donde la reutilización de modelos y sesiones es frecuente, sin elevar demasiado el riesgo de alcanzar el umbral agresivo.
*   Los demás umbrales y la configuración de concurrencia se mantuvieron en sus valores iniciales (o 'auto') ya que las pruebas no indicaron problemas significativos o áreas claras de mejora simple con ellos.

## 6. Conclusiones Finales de la Fase 6

El proceso de pruebas exhaustivas (unitarias, integración, carga) y benchmarking ha validado la funcionalidad y la interacción del nuevo sistema de gestión de recursos centralizado. Las pruebas de carga demostraron que el sistema es estable bajo estrés simulado y que el `ResourceManager` responde a los cambios en el uso de recursos según lo configurado. El benchmarking proporcionó métricas de rendimiento base. La optimización final de parámetros en `config.yaml` refleja un ajuste basado en las observaciones de estas pruebas, buscando un equilibrio entre reactividad, eficiencia y estabilidad. El sistema de gestión de recursos se considera funcional y configurado para un rendimiento razonable, listo para la documentación final (Fase 7). 