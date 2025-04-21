// Variables globales
let currentSession = null;
let isWaitingForResponse = false;
const API_BASE_URL = window.location.origin;

// Definir estilos CSS para mejorar la visualización de los botones de contexto
const styleSheet = document.createElement("style");
styleSheet.textContent = `
    .context-toggle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-left: 4px;
        cursor: pointer;
        background: transparent;
        border: none;
        color: var(--text-muted);
        padding: 4px;
        border-radius: 4px;
    }

    .context-toggle:hover {
        background-color: var(--hover-bg);
        color: var(--accent-color);
    }

    .message-controls {
        display: flex;
        align-items: center;
        margin-top: 6px;
    }

    .context-content {
        margin-top: 8px;
        padding: 10px;
        border-radius: 6px;
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
    }
`;
document.head.appendChild(styleSheet);

// Inicializar la interfaz
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM cargado, inicializando aplicación...');
    
    // Elementos DOM
    const chatContainer = document.getElementById('chat-container');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const clearButton = document.getElementById('clear-chat');
    const settingsButton = document.getElementById('settings-button') || document.getElementById('input-settings-button');
    const settingsModal = document.getElementById('settings-modal');
    const closeButtons = document.querySelectorAll('.close-button');
    const themeSelect = document.getElementById('theme-select');
    const modalThemeSelect = document.getElementById('modal-theme-select');
    const showSources = document.getElementById('show-sources');
    const messageDensity = document.getElementById('message-density');
    const messageDensityValue = document.getElementById('message-density-value');
    const reducedMotion = document.getElementById('reduced-motion');
    const highContrast = document.getElementById('high-contrast');
    const resetSettings = document.getElementById('reset-settings');
    const saveSettings = document.getElementById('save-settings');
    const databaseSelect = document.getElementById('database');
    const welcomeScreen = document.getElementById('welcome-screen');
    const exportChatBtn = document.getElementById('export-chat') || document.getElementById('input-export-chat');
    const charCounter = document.getElementById('char-counter');
    const themePreview = document.getElementById('theme-preview');
    const fontSizeSelect = document.getElementById('font-size');
    const showTimestampsCheck = document.getElementById('show-timestamps');
    const showAvatarsCheck = document.getElementById('show-avatars');

    // Validar elementos críticos
    if (!chatContainer) console.error('Elemento crítico no encontrado: chat-container');
    if (!chatInput) console.error('Elemento crítico no encontrado: chat-input');
    if (!sendButton) console.error('Elemento crítico no encontrado: send-button');
    if (!databaseSelect) console.error('Elemento crítico no encontrado: database');

    // Variables de estado
    let contextVisible = true;
    let currentDbInfo = null;
    let settings = {
        theme: localStorage.getItem('theme') || 'dark',
        showSources: localStorage.getItem('showSources') !== 'false',
        autoShowSources: localStorage.getItem('autoShowSources') === 'true',
        messageDensity: parseInt(localStorage.getItem('messageDensity') || 2),
        reducedMotion: localStorage.getItem('reducedMotion') === 'true',
        highContrast: localStorage.getItem('highContrast') === 'true',
        fontSize: localStorage.getItem('fontSize') || 'medium',
        showTimestamps: localStorage.getItem('showTimestamps') !== 'false',
        showAvatars: localStorage.getItem('showAvatars') !== 'false'
    };
    
    // Almacenamiento local de contextos para evitar problemas de 404
    const localContextStorage = {};

    // Configurar marked.js para procesar Markdown de forma segura
    if (typeof marked !== 'undefined') {
        // Crear un renderizador personalizado
        const customRenderer = new marked.Renderer();
        
        // Personalizar listas para hacerlas más compactas
        customRenderer.list = function(body, ordered, start) {
            const type = ordered ? 'ol' : 'ul';
            const startAttr = (ordered && start !== 1) ? ` start="${start}"` : '';
            return `<${type}${startAttr} class="compact-list">${body}</${type}>\n`;
        };
        
        // Personalizar bloques de código para asegurar que siempre se muestran bien
        customRenderer.code = function(code, infostring, escaped) {
            const lang = (infostring || '').match(/\S*/)[0];
            if (this.options.highlight) {
                const out = this.options.highlight(code, lang);
                if (out != null && out !== code) {
                    escaped = true;
                    code = out;
                }
            }
            
            if (!lang) {
                return `<pre><code class="hljs">${escaped ? code : escapeHTML(code)}</code></pre>`;
            }
            
            return `<pre><code class="hljs language-${escapeHTML(lang)}">${escaped ? code : escapeHTML(code)}</code></pre>`;
        };
        
        // Simplificar algunos elementos para visualización más clara
        customRenderer.table = function(header, body) {
            return `<div class="table-wrapper"><table class="compact-table">\n<thead>\n${header}</thead>\n<tbody>\n${body}</tbody>\n</table></div>\n`;
        };
        
        // Configurar opciones de marcado
        marked.setOptions({
            renderer: customRenderer,
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.error('Error al resaltar código:', err);
                    }
                }
                return code;
            },
            pedantic: false,
            gfm: true,
            breaks: true,
            sanitize: false,
            smartypants: false,
            xhtml: false
        });
    }

    // Almacenamiento en caché para el contexto de los mensajes
    const messageContextStore = new Map();

    // Función para escapar HTML y prevenir XSS
    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // Función auxiliar para desplazarse al final del chat
    function scrollToBottom() {
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    // Función para truncar texto para dispositivos pequeños
    function truncateTextForSmallDevices(text, maxLength = 100, isSystemMessage = false) {
        // Si estamos en una pantalla pequeña y el texto es más largo que maxLength
        if (window.innerWidth <= 480 && text.length > maxLength) {
            if (isSystemMessage) {
                // Para mensajes del sistema, usamos un maxLength más corto en dispositivos muy pequeños
                if (window.innerWidth <= 360) {
                    maxLength = 70;
                }
                return text.substring(0, maxLength) + "...";
            }
            return text.substring(0, maxLength) + "...";
        }
        return text;
    }

    // Función mejorada para ajustar altura del textarea automáticamente
    function adjustTextareaHeight(textarea) {
        textarea.style.height = 'auto';
        const maxHeight = window.innerHeight * 0.3; // Limitar altura máxima a 30% de la ventana
        if (textarea.scrollHeight > maxHeight) {
            textarea.style.height = maxHeight + "px";
            textarea.style.overflowY = "scroll";
        } else {
            textarea.style.overflowY = "hidden";
        }
    }

    // Función para manejar la visualización en dispositivos móviles
    function setupMobileView() {
        const isMobile = window.innerWidth <= 768;
        const isVerySmall = window.innerWidth <= 360;
        
        if (isVerySmall && !document.body.classList.contains('very-small-device')) {
            document.body.classList.add('very-small-device');
        } else if (!isVerySmall && document.body.classList.contains('very-small-device')) {
            document.body.classList.remove('very-small-device');
        }
        
        if (isMobile && !document.body.classList.contains('mobile-view')) {
            document.body.classList.add('mobile-view');
            
            // Reducir la cantidad de fuentes mostradas en móvil
            document.querySelectorAll('.context-content').forEach(context => {
                const maxItems = isVerySmall ? 2 : 3; // En dispositivos muy pequeños, mostrar menos items
                if (context.querySelectorAll('.context-item').length > maxItems) {
                    const items = Array.from(context.querySelectorAll('.context-item'));
                    items.slice(maxItems).forEach(item => item.classList.add('mobile-hidden'));
                    
                    if (!context.querySelector('.show-more-btn')) {
                        const showMoreBtn = document.createElement('button');
                        showMoreBtn.className = 'show-more-btn';
                        showMoreBtn.textContent = 'Mostrar más fuentes';
                        showMoreBtn.addEventListener('click', function() {
                            context.querySelectorAll('.mobile-hidden').forEach(item => {
                                item.classList.remove('mobile-hidden');
                            });
                            this.remove();
                        });
                        context.appendChild(showMoreBtn);
                    }
                }
            });
            
            // Ajustar mensajes del sistema para dispositivos móviles
            document.querySelectorAll('.system-message .message-content').forEach(msg => {
                const originalText = msg.getAttribute('data-original-text') || msg.textContent;
                if (!msg.getAttribute('data-original-text')) {
                    msg.setAttribute('data-original-text', originalText);
                }
                
                if (isVerySmall) {
                    msg.textContent = truncateTextForSmallDevices(originalText, 80);
                } else {
                    msg.textContent = truncateTextForSmallDevices(originalText);
                }
            });
            
        } else if (!isMobile && document.body.classList.contains('mobile-view')) {
            document.body.classList.remove('mobile-view');
            document.querySelectorAll('.mobile-hidden').forEach(item => {
                item.classList.remove('mobile-hidden');
            });
            document.querySelectorAll('.show-more-btn').forEach(btn => btn.remove());
            
            // Restaurar texto original en mensajes del sistema
            document.querySelectorAll('.system-message .message-content').forEach(msg => {
                const originalText = msg.getAttribute('data-original-text');
                if (originalText) {
                    msg.textContent = originalText;
                }
            });
        }
        
        // Ajustar altura del textarea para diferentes dispositivos
        adjustTextareaHeight(chatInput);
        
        // Verificar si es un dispositivo muy pequeño
        if (window.innerWidth <= 360) {
            document.body.classList.add('very-small-device');
            
            // Reducir el número de elementos de contexto mostrados
            const visibleContextItems = 2;
            const contextItems = document.querySelectorAll('.context-item');
            if (contextItems.length > visibleContextItems) {
                for (let i = visibleContextItems; i < contextItems.length; i++) {
                    contextItems[i].style.display = 'none';
                }
            }
        } else if (window.innerWidth <= 480) {
            // Para dispositivos pequeños pero no muy pequeños
            const visibleContextItems = 3;
            const contextItems = document.querySelectorAll('.context-item');
            if (contextItems.length > visibleContextItems) {
                for (let i = visibleContextItems; i < contextItems.length; i++) {
                    contextItems[i].style.display = 'none';
                }
            }
        }
        
        // Detectar dispositivos con memoria limitada
        if ('deviceMemory' in navigator) {
            // Si el dispositivo tiene menos de 4GB de RAM, reducir animaciones
            if (navigator.deviceMemory < 4) {
                document.body.classList.add('reduce-motion');
            }
        }
    }

    // Inicializar configuraciones
    function initSettings() {
        // Inicializar controles con los valores guardados
        if (showSources) {
            showSources.checked = settings.showSources;
        }
        
        if (document.getElementById('auto-show-sources')) {
            document.getElementById('auto-show-sources').checked = settings.autoShowSources;
        }
        
        if (messageDensity) {
            messageDensity.value = settings.messageDensity;
            updateDensityLabel();
        }

        // Cargar configuraciones guardadas
        fontSizeSelect.value = settings.fontSize;
        highContrast.checked = settings.highContrast;
        showTimestampsCheck.checked = settings.showTimestamps;
        showAvatarsCheck.checked = settings.showAvatars;

        // Aplicar configuraciones
        document.body.classList.toggle('font-small', settings.fontSize === 'small');
        document.body.classList.toggle('font-large', settings.fontSize === 'large');
        document.body.classList.toggle('high-contrast', settings.highContrast);
        document.body.classList.toggle('hide-timestamps', !settings.showTimestamps);
        document.body.classList.toggle('hide-avatars', !settings.showAvatars);

        // Event listeners para configuraciones
        fontSizeSelect.addEventListener('change', () => {
            settings.fontSize = fontSizeSelect.value;
            document.body.classList.remove('font-small', 'font-large');
            if (settings.fontSize !== 'medium') {
                document.body.classList.add(`font-${settings.fontSize}`);
            }
            localStorage.setItem('fontSize', settings.fontSize);
        });

        highContrast.addEventListener('change', () => {
            settings.highContrast = highContrast.checked;
            document.body.classList.toggle('high-contrast', settings.highContrast);
            localStorage.setItem('highContrast', settings.highContrast);
        });

        showTimestampsCheck.addEventListener('change', () => {
            settings.showTimestamps = showTimestampsCheck.checked;
            document.body.classList.toggle('hide-timestamps', !settings.showTimestamps);
            localStorage.setItem('showTimestamps', settings.showTimestamps);
        });

        showAvatarsCheck.addEventListener('change', () => {
            settings.showAvatars = showAvatarsCheck.checked;
            document.body.classList.toggle('hide-avatars', !settings.showAvatars);
            localStorage.setItem('showAvatars', settings.showAvatars);
        });
    }

    // Panel de configuración
    settingsButton.addEventListener('click', () => {
        // Update modal fields with current settings
        modalThemeSelect.value = settings.theme;
        showSources.checked = settings.showSources;
        messageDensity.value = settings.messageDensity;
        updateDensityLabel();
        reducedMotion.checked = settings.reducedMotion;
        highContrast.checked = settings.highContrast;
        
        settingsModal.classList.add('active');
    });

    // Close modal
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            settingsModal.classList.remove('active');
        });
    });

    // Close modal on outside click
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.remove('active');
        }
    });

    // Theme selector in header
    themeSelect.addEventListener('change', () => {
        settings.theme = themeSelect.value;
        applyTheme();
        saveSettings();
    });

    // Message density slider
    messageDensity.addEventListener('input', updateDensityLabel);

    // Reset settings
    resetSettings.addEventListener('click', () => {
        settings = {
            theme: 'dark',
            showSources: true,
            autoShowSources: false,
            messageDensity: 2,
            reducedMotion: false,
            highContrast: false,
            fontSize: 'medium',
            showTimestamps: true,
            showAvatars: true
        };
        
        // Update UI
        modalThemeSelect.value = settings.theme;
        showSources.checked = settings.showSources;
        messageDensity.value = settings.messageDensity;
        updateDensityLabel();
        reducedMotion.checked = settings.reducedMotion;
        highContrast.checked = settings.highContrast;
        fontSizeSelect.value = settings.fontSize;
        showTimestampsCheck.checked = settings.showTimestamps;
        showAvatarsCheck.checked = settings.showAvatars;
    });

    // Save settings and close modal
    if (saveSettings) {
        saveSettings.addEventListener('click', () => {
            // Update settings based on modal inputs
            if (modalThemeSelect) settings.theme = modalThemeSelect.value;
            if (showSources) settings.showSources = showSources.checked;
            if (document.getElementById('auto-show-sources')) settings.autoShowSources = document.getElementById('auto-show-sources').checked;
            if (messageDensity) settings.messageDensity = parseInt(messageDensity.value);
            if (reducedMotion) settings.reducedMotion = reducedMotion.checked;
            if (highContrast) settings.highContrast = highContrast.checked;
            if (fontSizeSelect) settings.fontSize = fontSizeSelect.value;
            if (showTimestampsCheck) settings.showTimestamps = showTimestampsCheck.checked;
            if (showAvatarsCheck) settings.showAvatars = showAvatarsCheck.checked;
            
            // Apply and save settings
            applySettings();
            saveSettingsToStorage();
            
            // Close modal
            settingsModal.classList.remove('active');
        });
    }

    // Apply all settings
    function applySettings() {
        applyTheme();
        applySourcesVisibility();
        applyMessageDensity();
        applyAccessibilitySettings();
        
        // Sync header theme selector with settings
        themeSelect.value = settings.theme;
    }

    // Apply theme
    function applyTheme() {
        setTheme(settings.theme);
    }

    // Set theme by adding the appropriate class to body
    function setTheme(theme) {
        // Remove existing theme classes
        document.body.classList.remove('light-theme', 'aqua-theme');
        
        // Add the selected theme class (dark theme is default, no class needed)
        if (theme === 'light') {
            document.body.classList.add('light-theme');
        } else if (theme === 'aqua') {
            document.body.classList.add('aqua-theme');
        }
        
        // Update theme selectors
        const themeSelectors = document.querySelectorAll('#theme-select, #modal-theme-select');
        themeSelectors.forEach(selector => {
            if (selector) selector.value = theme;
        });
        
        // Save to settings
        settings.theme = theme;
    }

    // Apply sources visibility
    function applySourcesVisibility() {
        const sourcesPanels = document.querySelectorAll('.sources-panel');
        sourcesPanels.forEach(panel => {
            panel.style.display = settings.showSources ? 'block' : 'none';
        });
        
        // Controlar visibilidad de botones de contexto
        document.querySelectorAll('.context-toggle').forEach(btn => {
            btn.style.display = settings.showSources ? 'inline-flex' : 'none';
        });
        
        // Si está activada la opción de mostrar fuentes automáticamente,
        // abrir automáticamente los paneles de contexto que están cerrados
        if (settings.autoShowSources && settings.showSources) {
            document.querySelectorAll('.context-content').forEach(container => {
                if (container.style.display === 'none' && container.id.startsWith('context-')) {
                    const messageId = container.id.replace('context-', '');
                    const button = document.querySelector(`.context-toggle[data-message-id="${messageId}"]`);
                    if (button) {
                        // Solo abrir si está cerrado y tiene botón asociado
                        fetchAndDisplayContext(messageId, button, container);
                    }
                }
            });
        }
    }

    // Apply message density
    function applyMessageDensity() {
        document.documentElement.style.setProperty('--message-spacing', 
            settings.messageDensity === 1 ? '8px' : 
            settings.messageDensity === 3 ? '24px' : '16px');
        
        document.documentElement.style.setProperty('--message-padding', 
            settings.messageDensity === 1 ? '8px' : 
            settings.messageDensity === 3 ? '16px' : '12px');
    }

    // Apply accessibility settings
    function applyAccessibilitySettings() {
        if (settings.reducedMotion) {
            document.body.classList.add('reduced-motion');
        } else {
            document.body.classList.remove('reduced-motion');
        }
        
        if (settings.highContrast) {
            document.body.classList.add('high-contrast');
        } else {
            document.body.classList.remove('high-contrast');
        }
    }

    // Update density label
    function updateDensityLabel() {
        const value = parseInt(messageDensity.value);
        messageDensityValue.textContent = 
            value === 1 ? 'Compacto' : 
            value === 3 ? 'Espaciado' : 'Normal';
    }

    // Save settings to localStorage
    function saveSettingsToStorage() {
        localStorage.setItem('theme', settings.theme);
        localStorage.setItem('showSources', settings.showSources);
        localStorage.setItem('autoShowSources', settings.autoShowSources);
        localStorage.setItem('messageDensity', settings.messageDensity);
        localStorage.setItem('reducedMotion', settings.reducedMotion);
        localStorage.setItem('highContrast', settings.highContrast);
        localStorage.setItem('fontSize', settings.fontSize);
        localStorage.setItem('showTimestamps', settings.showTimestamps);
        localStorage.setItem('showAvatars', settings.showAvatars);
    }

    // Función para cargar la lista de bases de datos disponibles
    async function loadDatabases() {
        try {
            console.log("Cargando bases de datos disponibles...");
            
            if (!databaseSelect) {
                console.error("Error: Elemento select de base de datos no encontrado");
                return;
            }
            
            databaseSelect.innerHTML = '<option value="" disabled selected>Cargando bases de datos...</option>';
            
            console.log("Realizando fetch a /api/databases...");
            const response = await fetch('/api/databases');
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error al cargar bases de datos:', response.status, errorText);
                throw new Error(`Error al cargar las bases de datos: ${response.status} ${response.statusText}`);
            }
            
            console.log("Respuesta recibida, procesando JSON...");
            const data = await response.json();
            console.log("Datos recibidos:", data);
            
            databaseSelect.innerHTML = '<option value="" disabled selected>Seleccione una base de datos</option>';
            
            let databaseCount = 0;
            
            if (data.database_details && Array.isArray(data.database_details)) {
                databaseCount = data.database_details.length;
                console.log(`Procesando ${databaseCount} bases de datos desde database_details`);
                
                data.database_details.forEach(db => {
                    const option = document.createElement('option');
                    option.value = db.id;
                    option.textContent = db.display_name || `${db.name} (${db.type})`;
                    option.dataset.type = db.type;
                    option.dataset.size = db.size_formatted || '';
                    databaseSelect.appendChild(option);
                });
            } else if (Array.isArray(data.databases)) {
                databaseCount = data.databases.length;
                console.log(`Procesando ${databaseCount} bases de datos desde databases`);
                
                data.databases.forEach(db => {
                    const option = document.createElement('option');
                    option.value = db;
                    option.textContent = db;
                    databaseSelect.appendChild(option);
                });
            } else {
                console.warn("Formato de respuesta inesperado:", data);
            }
            
            if (databaseCount === 0) {
                console.log("No se encontraron bases de datos");
                addSystemMessage('No hay bases de datos disponibles. Por favor, ingesta documentos primero.');
                databaseSelect.innerHTML = '<option value="" disabled selected>No hay bases de datos disponibles</option>';
            } else {
                console.log(`Bases de datos cargadas: ${databaseCount}`);
                addSystemMessage(`Se han cargado ${databaseCount} bases de datos. Seleccione una para iniciar el chat.`);
            }
        } catch (error) {
            console.error('Error al cargar bases de datos:', error);
            
            if (!databaseSelect) {
                console.error("Error adicional: Elemento select de base de datos no encontrado");
                return;
            }
            
            databaseSelect.innerHTML = '<option value="" disabled selected>Error al cargar bases de datos</option>';
            
            // Intentar agregar un mensaje al sistema
            try {
                addSystemMessage('Error al cargar las bases de datos. Por favor, recargue la página o verifique la conexión al servidor.');
            } catch (msgError) {
                console.error('Error adicional al mostrar mensaje:', msgError);
            }
            
            // Intentar cargar de nuevo después de un retraso
            setTimeout(() => {
                console.log("Reintentando carga de bases de datos...");
                try {
                    loadDatabases();
                } catch (retryError) {
                    console.error('Error en reintento de carga:', retryError);
                }
            }, 5000);
        }
    }

    // Iniciar sesión al seleccionar una base de datos
    databaseSelect.addEventListener('change', async () => {
        const selectedDb = databaseSelect.value;
        if (!selectedDb) return;
        
        try {
            addSystemMessage(`Iniciando sesión con la base de datos: ${selectedDb}...`);
            chatInput.disabled = true;
            sendButton.disabled = true;

            console.log(`Creando sesión para base de datos: ${selectedDb}`);
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ database_name: selectedDb })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error en la respuesta del servidor:', response.status, errorText);
                throw new Error(`Error al crear la sesión: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('Sesión creada:', data);

            currentSession = data.session_id;
            currentDbInfo = data.db_info;
            const dbType = data.db_type || databaseSelect.selectedOptions[0].dataset.type || 'desconocido';

            // Ocultar pantalla de bienvenida
            welcomeScreen.classList.add('hidden');
            
            // Limpiar chat y mostrar mensaje inicial
            chatContainer.innerHTML = '';
            
            // Mensaje inicial simplificado para dispositivos pequeños
            let initialMessage;
            
            if (window.innerWidth <= 360) {
                // Mensaje muy reducido para dispositivos muy pequeños
                initialMessage = "¡Hola! Soy tu asistente IA. Puedo ayudarte a encontrar información en los documentos disponibles.";
            } else if (window.innerWidth <= 480) {
                // Mensaje reducido para dispositivos pequeños
                initialMessage = "¡Hola! Soy tu asistente IA. Puedo responder preguntas basadas en los documentos disponibles. ¿En qué puedo ayudarte hoy?";
            } else {
                // Mensaje completo para dispositivos normales
                initialMessage = "¡Hola! Soy tu asistente IA. Puedo responder a tus preguntas basadas en la información de los documentos disponibles. Puedes preguntarme sobre cualquier tema relacionado con estos documentos y trataré de ayudarte. ¿En qué puedo ayudarte hoy?";
            }
            
            // Agregar el mensaje inicial como un mensaje del sistema
            addBotMessage(initialMessage, true);
            
            // Mensaje de ayuda adaptado para tamaño de pantalla
            const helpMessage = window.innerWidth <= 360 
                ? '¿En qué puedo ayudarte?'
                : '¿En qué puedo ayudarte? Puedes hacerme preguntas sobre los documentos almacenados en esta base de datos.';
            
            addSystemMessage(helpMessage);

            // Habilitar controles
            chatInput.disabled = false;
            sendButton.disabled = false;
            clearButton.disabled = false;
            exportChatBtn.disabled = false;
            
            // Enfocar en el área de entrada
            chatInput.focus();
        } catch (error) {
            console.error('Error al iniciar sesión:', error);
            addSystemMessage(`Error al iniciar la sesión: ${error.message}. Por favor, intente de nuevo.`);
            // Re-habilitar la selección de base de datos
            databaseSelect.disabled = false;
        }
    });

    // Contador de caracteres
    chatInput.addEventListener('input', () => {
        const count = chatInput.value.length;
        if (charCounter) {
            charCounter.textContent = count;
            charCounter.classList.toggle('limit-near', count > 500);
            charCounter.classList.toggle('limit-reached', count > 1000);
        }
        
        // Ajustar altura del textarea
        adjustTextareaHeight(chatInput);
    });

    // Botón de limpiar chat
    clearButton.addEventListener('click', () => {
        if (confirm('¿Estás seguro de que deseas limpiar la conversación actual?')) {
            chatContainer.innerHTML = '';
            addSystemMessage('La conversación ha sido reiniciada. Puedes realizar nuevas consultas.');
            chatInput.focus();
        }
    });

    // Botón de exportar chat
    exportChatBtn.addEventListener('click', () => {
        const messages = chatContainer.querySelectorAll('.message');
        let exportText = 'Conversación RAG Chatbot\n';
        exportText += `Fecha: ${new Date().toLocaleString()}\n`;
        exportText += `Base de datos: ${databaseSelect.selectedOptions[0].textContent}\n\n`;
        
        messages.forEach(msg => {
            const isUser = msg.classList.contains('user-message');
            const content = msg.querySelector('.message-content');
            if (content) {
                const text = content.getAttribute('data-original-text') || content.textContent;
                exportText += `${isUser ? 'Usuario' : 'Bot'}: ${text}\n\n`;
            }
        });
        
        const blob = new Blob([exportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-export-${new Date().toISOString().slice(0, 10)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Manejo del envío de mensajes
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        } else if (event.key === 'Enter' && event.shiftKey) {
            // Permitir nueva línea con Shift+Enter
            const cursorPos = chatInput.selectionStart;
            const textBefore = chatInput.value.substring(0, cursorPos);
            const textAfter = chatInput.value.substring(cursorPos);
            chatInput.value = textBefore + '\n' + textAfter;
            chatInput.selectionStart = chatInput.selectionEnd = cursorPos + 1;
            adjustTextareaHeight(chatInput);
            event.preventDefault();
        }
    });

    // Send a message
    async function sendMessage() {
        if (!chatInput || !currentSession || isWaitingForResponse) return;
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        console.log("Estado de la sesión antes de enviar mensaje:", {
            sessionId: currentSession,
            isWaitingForResponse: isWaitingForResponse
        });
        
        isWaitingForResponse = true;
        addUserMessage(message);
        
        if (chatInput) {
            chatInput.value = '';
            chatInput.style.height = 'auto'; // Resetear altura
        }
        
        if (charCounter) charCounter.textContent = '0';
        
        const botMessageElement = addBotMessage('', true);
        if (!botMessageElement) {
            isWaitingForResponse = false;
            console.error("No se pudo crear el elemento para el mensaje del bot");
            return;
        }

        try {
            // Mostrar indicador de "escribiendo..."
            const typingIndicator = botMessageElement.querySelector('.typing-indicator');
            if (typingIndicator) typingIndicator.style.display = 'flex';

            // Datos que enviaremos al servidor
            const queryData = { 
                session_id: currentSession, 
                query: message, 
                stream: true 
            };
            
            // Realizar la consulta al servidor
            console.log("Enviando consulta a API:", {
                endpoint: '/api/query',
                data: queryData,
                message: message
            });
            
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(queryData)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error en la respuesta del servidor:', {
                    status: response.status,
                    statusText: response.statusText,
                    errorBody: errorText
                });
                throw new Error(`Error al enviar la consulta: ${response.status} ${response.statusText}`);
            }

            // Verificar que el cuerpo de la respuesta sea válido
            if (!response.body) {
                throw new Error("La respuesta no contiene un cuerpo de datos");
            }

            // Procesar respuesta streaming
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let lastMessageId = null;
            let chunkCount = 0;

            // Intentar obtener el contexto de manera asíncrona
            let contextData = [];
            try {
                // Ejecutar la consulta sin streaming para obtener el contexto
                const contextResponse = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...queryData, stream: false })
                });
                
                if (contextResponse.ok) {
                    const contextResult = await contextResponse.json();
                    if (contextResult.context && Array.isArray(contextResult.context)) {
                        contextData = contextResult.context;
                        console.log(`Contexto obtenido para consulta: ${contextData.length} fragmentos`);
                    }
                }
            } catch (contextError) {
                console.warn('Error al obtener contexto anticipado:', contextError);
                // No hacer nada, seguiremos con el streaming normal
            }

            // Leer el stream de respuesta
            console.log('Procesando respuesta streaming...');
            while (true) {
                try {
                    const { done, value } = await reader.read();
                    if (done) {
                        console.log("Stream de respuesta completado después de", chunkCount, "chunks");
                        // Verificar si hay un mensaje ID en el texto completo antes de terminar
                        const finalIdMatch = fullText.match(/<hidden_message_id>([^<]+)<\/hidden_message_id>/);
                        if (finalIdMatch && !lastMessageId) {
                            const messageId = finalIdMatch[1];
                            console.log(`ID de mensaje encontrado en texto completo: ${messageId}`);
                            fullText = fullText.replace(/<hidden_message_id>[^<]+<\/hidden_message_id>/, '');
                            
                            // Guardar el contexto localmente si no lo hemos hecho ya
                            if (contextData.length > 0) {
                                localContextStorage[messageId] = {
                                    query: message,
                                    context: contextData,
                                    timestamp: new Date().toISOString()
                                };
                                console.log(`Contexto guardado localmente para mensaje final ${messageId}: ${contextData.length} fragmentos`);
                            }
                            
                            // Actualizar mensaje con ID
                            updateBotMessage(botMessageElement, fullText, messageId);
                            console.log(`Mensaje actualizado con ID final: ${messageId}`);
                        } else if (!lastMessageId) {
                            // Si llegamos al final sin encontrar un ID de mensaje, generamos uno
                            const fallbackId = `msg_${Date.now()}`;
                            console.log(`No se detectó ID en la respuesta. Usando ID generado: ${fallbackId}`);
                            
                            if (contextData.length > 0) {
                                localContextStorage[fallbackId] = {
                                    query: message,
                                    context: contextData,
                                    timestamp: new Date().toISOString()
                                };
                                console.log(`Contexto guardado localmente con ID generado: ${contextData.length} fragmentos`);
                            }
                            
                            updateBotMessage(botMessageElement, fullText, fallbackId);
                        }
                        break;
                    }
                    
                    chunkCount++;
                    const chunk = decoder.decode(value, { stream: true });
                    if (!chunk) {
                        console.warn("Chunk recibido vacío, continuando...");
                        continue;
                    }

                    console.log(`Chunk #${chunkCount} recibido (${chunk.length} bytes)`, { chunkPreview: chunk.substring(0, 50) });
                    
                    // Procesar los chunks en formato SSE (Server-Sent Events)
                    const lines = chunk.split('\n');
                    let processedText = '';
                    
                    for (const line of lines) {
                        // Ignorar líneas vacías
                        if (!line.trim()) continue;
                        
                        // Ignorar el marcador [DONE]
                        if (line.includes('[DONE]')) continue;
                        
                        // Extraer el contenido eliminando el prefijo "data: "
                        if (line.startsWith('data: ')) {
                            const textContent = line.substring(6); // Eliminar "data: "
                            processedText += textContent;
                        } else {
                            // Si no tiene el formato esperado, usar el texto tal cual
                            processedText += line;
                        }
                    }
                    
                    // Manejar ID oculto - buscar en diferentes formatos
                    let messageId = null;
                    
                    // Formato 1: <hidden_message_id>ID</hidden_message_id>
                    let hiddenMatch = processedText.match(/<hidden_message_id>([^<]+)<\/hidden_message_id>/);
                    if (hiddenMatch) {
                        messageId = hiddenMatch[1];
                        console.log(`ID de mensaje encontrado (formato hidden): ${messageId}`);
                        processedText = processedText.replace(/<hidden_message_id>[^<]+<\/hidden_message_id>/, '');
                    } 
                    // Formato 2: <message-id:ID>
                    else {
                        let oldMatch = processedText.match(/<message-id:([^>]+)>/);
                        if (oldMatch) {
                            messageId = oldMatch[1];
                            console.log(`ID de mensaje encontrado (formato antiguo): ${messageId}`);
                            processedText = processedText.replace(/<message-id:[^>]+>/, '');
                        }
                    }
                    
                    // Acumular texto procesado
                    fullText += processedText;
                    
                    // Actualizar el mensaje en curso
                    updateBotMessage(botMessageElement, fullText);
                    
                    // Si se encontró un messageId, finalizar procesamiento
                    if (messageId) {
                        console.log(`Procesando mensaje completo con ID: ${messageId}`);
                        
                        // Guardar el contexto para este mensaje localmente
                        if (contextData.length > 0) {
                            localContextStorage[messageId] = {
                                query: message,
                                context: contextData,
                                timestamp: new Date().toISOString()
                            };
                            console.log(`Contexto guardado localmente: ${contextData.length} fragmentos`);
                            
                            // Mostrar detalles del primer fragmento para depuración
                            if (contextData[0]) {
                                console.log("Muestra del primer fragmento:", {
                                    header: contextData[0].header,
                                    text_length: contextData[0].text ? contextData[0].text.length : 0,
                                    document: contextData[0].document
                                });
                            }
                        } else {
                            console.warn("No hay datos de contexto para guardar con este mensaje");
                        }
                        
                        // Actualizar mensaje con ID y guardar texto final
                        updateBotMessage(botMessageElement, fullText, messageId);
                        
                        // Almacenar mensaje ID localmente para depuración
                        lastMessageId = messageId;
                        
                        break; // Terminar el ciclo ya que tenemos el mensaje completo con ID
                    }
                } catch (chunkError) {
                    console.error("Error al procesar chunk:", chunkError);
                    // Continuar leyendo chunks a pesar del error
                }
            }
        } catch (error) {
            console.error('Error durante la consulta:', error);
            updateBotMessage(botMessageElement, `Error al procesar la consulta: ${error.message}`);
            
            // Intentar recuperar la sesión si parece que se ha perdido
            if (error.message.includes("Sesión no válida") || error.message.includes("expirada")) {
                console.warn("Posible error de sesión. Intentando recuperar estado...");
                // Sugerir al usuario que recargue la página si la sesión expiró
                addSystemMessage("Parece que la sesión ha expirado. Por favor, seleccione la base de datos nuevamente o recargue la página.");
            }
        } finally {
            isWaitingForResponse = false;
            chatInput.focus();
            setupMobileView(); // Ajustar vista al terminar de recibir respuesta
        }
    }

    // Función para limpiar el texto de patrones Markdown no deseados
    function cleanMarkdownText(text) {
        if (!text) return '';
        
        // Eliminar líneas de separación (patrones como {2}------------------------)
        let cleaned = text.replace(/\{[\d]*\}[-]+/g, '');
        
        // Eliminar líneas horizontales (---, ___, ***)
        cleaned = cleaned.replace(/^[-_*]{3,}$/gm, '');
        
        // Eliminar marcadores de comienzo y fin de código cuando no están correctamente formateados
        cleaned = cleaned.replace(/^```\s*(\w+)?$/gm, '');
        
        // Eliminar caracteres de escape innecesarios
        cleaned = cleaned.replace(/\\([^\\])/g, '$1');
        
        // Eliminar líneas vacías múltiples (dejar solo una)
        cleaned = cleaned.replace(/\n{3,}/g, '\n\n');

        // Eliminar etiquetas innecesarias que pueden aparecer en el texto
        cleaned = cleaned.replace(/<\/?p>/g, '');
        cleaned = cleaned.replace(/<br\s*\/?>/g, '\n');
        
        // Eliminar comentarios HTML
        cleaned = cleaned.replace(/<!--[\s\S]*?-->/g, '');
        
        // Simplificar cadenas de caracteres especiales repetidos (##, **, etc.)
        cleaned = cleaned.replace(/#{3,}/g, '## ');
        cleaned = cleaned.replace(/\*{3,}/g, '**');
        cleaned = cleaned.replace(/_{3,}/g, '__');
        
        // Eliminar símbolos del inicio de líneas que no tienen sentido
        cleaned = cleaned.replace(/^[>\s]*([^a-zA-Z0-9<])\s*/gm, '$1 ');
        
        // Eliminar notación de referencia [1], [2], etc. cuando aparece sola en una línea
        cleaned = cleaned.replace(/^\s*\[\d+\]\s*$/gm, '');
        
        // Eliminar líneas que solo contienen números o símbolos de puntuación
        cleaned = cleaned.replace(/^\s*[\d.,;:!?]+\s*$/gm, '');
        
        return cleaned.trim();
    }
    
    // Mostrar/ocultar contexto
    async function fetchAndDisplayContext(messageId, contextButton, contextContainer) {
        if (contextContainer.style.display === 'block') {
            contextContainer.style.display = 'none';
            contextButton.textContent = 'Ver fuentes';
            return;
        }
        
        // Mostrar indicador de carga
        contextContainer.innerHTML = '<div class="context-loading"><span class="material-symbols-outlined">hourglass_empty</span> Cargando fuentes...</div>';
        contextContainer.style.display = 'block';
        contextButton.textContent = 'Ocultar fuentes';
        
        // Verificar si hay sesión activa
        if (!currentSession) {
            console.error('Error: No hay sesión activa para recuperar contexto');
            contextContainer.innerHTML = `
                <div class="context-error">
                    <div class="error-icon"><span class="material-symbols-outlined">error</span></div>
                    <div class="error-content">
                        <h4>Error de sesión</h4>
                        <p>No hay una sesión activa. Recargue la página e intente nuevamente.</p>
                    </div>
                </div>
            `;
            return;
        }
        
        console.log(`Solicitando contexto para mensaje ID: ${messageId}, Sesión: ${currentSession}`);
        
        // Primero intentar usar el contexto almacenado localmente
        if (localContextStorage[messageId]) {
            try {
                console.log(`Usando contexto almacenado localmente para mensaje ${messageId}`);
                const localData = localContextStorage[messageId];
                
                // Verificar si hay contexto
                if (!localData.context || !localData.context.length) {
                    contextContainer.innerHTML = `
                        <div class="context-empty">
                            <div class="info-icon"><span class="material-symbols-outlined">source</span></div>
                            <div class="info-content">
                                <h4>Sin fuentes disponibles</h4>
                                <p>Esta respuesta fue generada sin referencias específicas a los documentos de la base de datos.</p>
                            </div>
                        </div>
                    `;
                    return;
                }
                
                // Construir el HTML para mostrar el contexto local
                displayContextData(localData.context, contextContainer);
                return;
            } catch (localError) {
                console.error('Error al usar contexto local:', localError);
                // Si falla, continuamos con la solicitud al servidor
            }
        }
        
        // Intentar recuperar el contexto del servidor con reintentos
        let attempts = 0;
        const maxAttempts = 3;
        let backoffDelay = 500; // ms
        
        while (attempts < maxAttempts) {
            try {
                attempts++;
                
                // Construir la URL con el ID del mensaje y la sesión
                const contextUrl = `/api/message-context/${messageId}?session_id=${currentSession}`;
                console.log(`Intento ${attempts}/${maxAttempts}: Consultando ${contextUrl}`);
                
                const response = await fetch(contextUrl);
                const status = response.status;
                
                console.log(`Respuesta obtenida con estado: ${status}`);
                
                // Manejar diferentes códigos de estado
                if (status === 404) {
                    // Si tenemos contexto local, usarlo como respaldo
                    if (localContextStorage[messageId]) {
                        console.log(`Usando contexto local como respaldo para mensaje ${messageId}`);
                        const localData = localContextStorage[messageId];
                        
                        if (localData.context && localData.context.length > 0) {
                            displayContextData(localData.context, contextContainer);
                            return;
                        }
                    }
                    
                    // Si no hay respaldo local, mostrar mensaje de error
                    contextContainer.innerHTML = `
                        <div class="context-error">
                            <div class="error-icon"><span class="material-symbols-outlined">info</span></div>
                            <div class="error-content">
                                <h4>No se encontraron fuentes</h4>
                                <p>No hay información adicional disponible para esta respuesta.</p>
                            </div>
                        </div>
                    `;
                    return;
                } else if (status >= 400 && status < 500) {
                    // Error de cliente (400-499)
                    throw new Error(`Error en solicitud: ${status}`);
                } else if (status >= 500) {
                    // Error de servidor (500+)
                    throw new Error(`Error en servidor: ${status}`);
                }
                
                if (!response.ok) {
                    throw new Error(`Error al obtener contexto: ${response.status} ${response.statusText}`);
                }
                
                // Procesar la respuesta
                const data = await response.json();
                
                // Verificar si hay contexto
                if (!data.context || !data.context.length) {
                    contextContainer.innerHTML = `
                        <div class="context-empty">
                            <div class="info-icon"><span class="material-symbols-outlined">source</span></div>
                            <div class="info-content">
                                <h4>Sin fuentes disponibles</h4>
                                <p>Esta respuesta fue generada sin referencias específicas a los documentos de la base de datos.</p>
                            </div>
                        </div>
                    `;
                    return;
                }
                
                // Construir el HTML para mostrar el contexto
                displayContextData(data.context, contextContainer);
                return;
                
            } catch (error) {
                console.error(`Error al obtener contexto (intento ${attempts}/${maxAttempts}):`, error);
                
                // Si tenemos contexto local y este es el último intento, usarlo como respaldo
                if (attempts >= maxAttempts && localContextStorage[messageId]) {
                    console.log(`Usando contexto local como última opción para mensaje ${messageId}`);
                    const localData = localContextStorage[messageId];
                    
                    if (localData.context && localData.context.length > 0) {
                        displayContextData(localData.context, contextContainer);
                        return;
                    }
                }
                
                if (attempts >= maxAttempts) {
                    // Si se agotan los reintentos, mostrar mensaje de error
                    contextContainer.innerHTML = `
                        <div class="context-error">
                            <div class="error-icon"><span class="material-symbols-outlined">error</span></div>
                            <div class="error-content">
                                <h4>Error al recuperar las fuentes</h4>
                                <p>${error.message}</p>
                                <button class="retry-btn"><span class="material-symbols-outlined">refresh</span> Reintentar</button>
                            </div>
                        </div>
                    `;
                    
                    // Añadir event listener al botón de reintento
                    const retryBtn = contextContainer.querySelector('.retry-btn');
                    if (retryBtn) {
                        retryBtn.addEventListener('click', () => {
                            fetchAndDisplayContext(messageId, contextButton, contextContainer);
                        });
                    }
                } else {
                    // Incrementar el delay antes del próximo intento (backoff exponencial)
                    backoffDelay *= 1.5;
                    
                    // Mostrar mensaje de reintento
                    contextContainer.innerHTML = `
                        <div class="context-loading">
                            <span class="material-symbols-outlined">sync</span>
                            Reintentando (${attempts}/${maxAttempts})...
                        </div>
                    `;
                    
                    // Esperar antes de reintentar
                    await new Promise(resolve => setTimeout(resolve, backoffDelay));
                }
            }
        }
    }

    // Función para mostrar los datos de contexto
    function displayContextData(contextData, container) {
        // Construir el HTML para mostrar el contexto
        let html = `
            <div class="context-header">
                <span class="material-symbols-outlined">menu_book</span>
                <span>Fuentes utilizadas (${contextData.length})</span>
            </div>
            <div class="context-items-container">
        `;
        
        // Ordenar los chunks por relevancia (descendente)
        const sortedData = [...contextData].sort((a, b) => {
            const relevanceA = a.relevance || 0;
            const relevanceB = b.relevance || 0;
            return relevanceB - relevanceA;
        });
        
        // Añadir cada fuente
        sortedData.forEach((chunk, index) => {
            const className = index >= 3 && window.innerWidth <= 768 ? 'context-item mobile-hidden' : 'context-item';
            const relevancePercentage = chunk.relevance || 100;
            
            // Determinar clase de relevancia para el color
            let relevanceClass = 'relevance-high';
            if (relevancePercentage < 85) relevanceClass = 'relevance-medium';
            if (relevancePercentage < 70) relevanceClass = 'relevance-low';
            
            // Limpiar texto antes del procesamiento Markdown
            const cleanedText = cleanMarkdownText(chunk.text);
            
            // Formatear el texto con Markdown si está disponible
            let renderedText = cleanedText;
            if (typeof marked !== 'undefined') {
                try {
                    renderedText = marked.parse(cleanedText);
                } catch (err) {
                    console.error('Error al renderizar Markdown:', err);
                    renderedText = escapeHTML(cleanedText).replace(/\n/g, '<br>');
                }
            } else {
                renderedText = escapeHTML(cleanedText).replace(/\n/g, '<br>');
            }
            
            const documentTitle = chunk.document_title || chunk.document || 'Documento sin título';
            const source = chunk.source || '';
            
            html += `
                <div class="${className}" data-relevance="${relevancePercentage}">
                    <div class="context-item-header">
                        <div class="context-item-title">
                            <h4>${escapeHTML(chunk.header || 'Sin título')}</h4>
                            ${documentTitle ? `<div class="context-document"><span class="material-symbols-outlined">description</span>${escapeHTML(documentTitle)}</div>` : ''}
                            ${source ? `<div class="context-source">${escapeHTML(source)}</div>` : ''}
                        </div>
                        <div class="context-relevance">
                            <span class="relevance-label ${relevanceClass}">${relevancePercentage}% relevante</span>
                            <div class="relevance-bar"><div class="relevance-fill ${relevanceClass}" style="width: ${relevancePercentage}%"></div></div>
                        </div>
                    </div>
                    <div class="context-item-content markdown-content">${renderedText}</div>
                    <div class="context-item-meta">
                        ${chunk.page && chunk.page !== 'N/A' ? `<span class="context-page"><span class="material-symbols-outlined">auto_stories</span>Página ${chunk.page}</span>` : ''}
                        ${chunk.url ? `<a href="${chunk.url}" target="_blank" class="context-link"><span class="material-symbols-outlined">link</span>Ver documento original</a>` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        // Añadir botón "Mostrar más" en móvil si hay más de 3 fuentes
        if (contextData.length > 3 && window.innerWidth <= 768) {
            html += `<button class="show-more-btn"><span class="material-symbols-outlined">expand_more</span>Mostrar más fuentes</button>`;
        }
        
        container.innerHTML = html;
        
        // Activar highlight.js para resaltado de código si está disponible
        if (typeof hljs !== 'undefined') {
            container.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
        
        // Añadir event listener al botón "Mostrar más"
        const showMoreBtn = container.querySelector('.show-more-btn');
        if (showMoreBtn) {
            showMoreBtn.addEventListener('click', function() {
                container.querySelectorAll('.mobile-hidden').forEach(item => {
                    item.classList.remove('mobile-hidden');
                });
                this.remove();
            });
        }
    }

    // Funciones auxiliares
    function addUserMessage(text) {
        const chatContainer = document.getElementById('chat-container');
        if (!chatContainer) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const div = document.createElement('div');
        div.className = 'message user-message';
        
        div.innerHTML = `
            <div class="message-content-container">
                <div class="message-bubble">
                    <div class="message-content">${escapeHTML(text).replace(/\n/g, '<br>')}</div>
                </div>
                <div class="message-avatar">
                    <span class="material-symbols-outlined">person</span>
                </div>
            </div>
            ${settings.showTimestamps ? `<div class="message-timestamp">${timestamp}</div>` : ''}
        `;
        
        chatContainer.appendChild(div);
        scrollToBottom();
    }

    // Función para agregar mensaje del bot
    function addBotMessage(message, isSystemMessage = false) {
        const chatContainer = document.getElementById('chat-container');
        if (!chatContainer) return null;
        
        try {
            const timestamp = new Date().toLocaleTimeString();
            const div = document.createElement('div');
            div.className = isSystemMessage ? 'message system-message' : 'message bot-message';
            
            if (isSystemMessage) {
                // Mensaje del sistema
                div.innerHTML = `
                    <div class="message-bubble">
                        <span class="material-symbols-outlined">info</span>
                        <span class="message-content" data-original-text="${escapeHTML(message)}">${escapeHTML(message)}</span>
                    </div>
                `;
            } else if (message === '') {
                // Mensaje vacío (estado de carga)
                div.innerHTML = `
                    <div class="message-content-container">
                        <div class="message-avatar">
                            <span class="material-symbols-outlined">smart_toy</span>
                        </div>
                        <div class="message-bubble">
                            <div class="typing-indicator">
                                <div class="dot"></div>
                                <div class="dot"></div>
                                <div class="dot"></div>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Mensaje normal del bot
                // Procesar Markdown si la biblioteca marked está disponible
                let contentHtml;
                if (typeof marked !== 'undefined') {
                    try {
                        contentHtml = marked.parse(message);
                    } catch (markdownError) {
                        console.error('Error al procesar Markdown:', markdownError);
                        contentHtml = escapeHTML(message).replace(/\n/g,'<br>');
                    }
                } else {
                    contentHtml = escapeHTML(message).replace(/\n/g,'<br>');
                }
                
                div.innerHTML = `
                    <div class="message-content-container">
                        <div class="message-avatar">
                            <span class="material-symbols-outlined">smart_toy</span>
                        </div>
                        <div class="message-bubble">
                            <div class="message-content markdown-content" data-original-text="${escapeHTML(message)}">${contentHtml}</div>
                            <div class="message-controls">
                                <button class="context-toggle" title="Ver fuentes" style="display: none;">
                                    <span class="material-symbols-outlined">menu_book</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    ${settings.showTimestamps ? `<div class="message-timestamp">${timestamp}</div>` : ''}
                    <div class="context-content" style="display: none;"></div>
                `;
                
                // Activar highlight.js para resaltado de código si está disponible
                if (typeof hljs !== 'undefined') {
                    setTimeout(() => {
                        div.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                        });
                    }, 0);
                }
                
                // Agregar event listener para el botón de contexto
                const contextButton = div.querySelector('.context-toggle');
                const contextContainer = div.querySelector('.context-content');
                
                if (contextButton && contextContainer) {
                    contextButton.addEventListener('click', function() {
                        const messageId = div.dataset.messageId;
                        if (messageId) {
                            fetchAndDisplayContext(messageId, contextButton, contextContainer);
                        } else {
                            console.error('No hay ID de mensaje para mostrar contexto');
                        }
                    });
                }
            }
            
            chatContainer.appendChild(div);
            scrollToBottom();
            return div;
        } catch (e) {
            console.error('Error al agregar mensaje del bot:', e);
            return null;
        }
    }

    // Función para agregar mensaje del sistema
    function addSystemMessage(message) {
        const chatContainer = document.getElementById('chat-container');
        if (!chatContainer) return;
        
        const div = document.createElement('div');
        div.className = 'message system-message';
        div.innerHTML = `
            <div class="message-bubble">
                <span class="material-symbols-outlined">info</span>
                <span class="message-content" data-original-text="${escapeHTML(message)}">${escapeHTML(message)}</span>
            </div>
        `;
        chatContainer.appendChild(div);
        scrollToBottom();
    }

    function updateBotMessage(messageElement, messageText, messageId = null) {
        if (!messageElement) return;
        
        // Actualizar el contenido del mensaje
        const contentElement = messageElement.querySelector('.message-content');
        if (contentElement) {
            // Formatear el texto con Markdown si está disponible
            let contentHtml;
            if (typeof marked !== 'undefined') {
                try {
                    contentHtml = marked.parse(messageText);
                } catch (markdownError) {
                    console.error('Error al procesar Markdown:', markdownError);
                    contentHtml = escapeHTML(messageText).replace(/\n/g,'<br>');
                }
            } else {
                contentHtml = escapeHTML(messageText).replace(/\n/g,'<br>');
            }
            
            contentElement.setAttribute('data-original-text', escapeHTML(messageText));
            contentElement.innerHTML = contentHtml;
            
            // Aplicar resaltado de código si está disponible
            if (typeof hljs !== 'undefined') {
                setTimeout(() => {
                    contentElement.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightElement(block);
                    });
                }, 0);
            }
        }
        
        // Buscar o crear el botón de contexto si no existe
        let contextButton = messageElement.querySelector('.context-toggle');
        let contextContainer = messageElement.querySelector('.context-content');
        
        // Si no existe el botón de contexto, crearlo
        if (!contextButton) {
            const messageBubble = messageElement.querySelector('.message-bubble');
            if (messageBubble) {
                // Buscar o crear controles de mensaje
                let messageControls = messageBubble.querySelector('.message-controls');
                if (!messageControls) {
                    messageControls = document.createElement('div');
                    messageControls.className = 'message-controls';
                    messageBubble.appendChild(messageControls);
                }
                
                // Crear botón de contexto
                contextButton = document.createElement('button');
                contextButton.className = 'context-toggle';
                contextButton.title = 'Ver fuentes';
                contextButton.style.display = 'none';
                contextButton.innerHTML = '<span class="material-symbols-outlined">menu_book</span>';
                messageControls.appendChild(contextButton);
            }
        }
        
        // Si no existe el contenedor de contexto, crearlo
        if (!contextContainer) {
            contextContainer = document.createElement('div');
            contextContainer.className = 'context-content';
            contextContainer.style.display = 'none';
            messageElement.appendChild(contextContainer);
        }
        
        // Agregar event listener al botón de contexto
        if (contextButton && contextContainer) {
            // Eliminar listeners existentes para evitar duplicados
            const newButton = contextButton.cloneNode(true);
            contextButton.parentNode.replaceChild(newButton, contextButton);
            contextButton = newButton;
            
            // Agregar nuevo listener
            contextButton.addEventListener('click', function() {
                if (messageId) {
                    fetchAndDisplayContext(messageId, contextButton, contextContainer);
                }
            });
        }
        
        // Si hay un ID de mensaje, actualizar el dataset y mostrar el botón de contexto
        if (messageId) {
            messageElement.dataset.messageId = messageId;
            
            // Mostrar el botón de contexto si existe
            if (contextButton) {
                contextButton.style.display = 'inline-flex';
            }
            
            // Ocultar el indicador de carga si existe
            const typingIndicator = messageElement.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.style.display = 'none';
            }
        }
    }

    // Función para obtener el ID de sesión actual
    function getChatSessionId() {
        return currentSession;
    }

    // Función para obtener el contexto de un mensaje desde la API
    function fetchMessageContext(messageId) {
        console.log(`Solicitando contexto para mensaje: ${messageId}`);
        
        // Construir URL con el ID de sesión si está disponible
        let url = `/api/message-context/${messageId}`;
        const sessionId = getChatSessionId();
        
        if (sessionId) {
            url += `?session_id=${sessionId}`;
        } else {
            console.warn('No hay ID de sesión disponible para la solicitud de contexto');
        }
        
        return fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error HTTP: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(`Contexto recibido para mensaje ${messageId}:`, data);
                if (data.status === 'success') {
                    return data;
                } else {
                    throw new Error(data.message || 'Error al obtener contexto');
                }
            });
    }

    // Función para cargar y mostrar el contexto de un mensaje
    function toggleMessageContext(messageId, contextContainer) {
        console.log(`Alternando visualización de contexto para mensaje: ${messageId}`);
        
        // Si el contenedor ya tiene contenido, alternamos su visibilidad
        if (contextContainer.innerHTML.trim() !== '') {
            const isVisible = contextContainer.style.display !== 'none';
            contextContainer.style.display = isVisible ? 'none' : 'block';
            console.log(`Contexto ${isVisible ? 'ocultado' : 'mostrado'}`);
            return;
        }
        
        // Mostrar indicador de carga
        contextContainer.innerHTML = '<div class="context-loading">Cargando fuentes...</div>';
        contextContainer.style.display = 'block';
        
        // Verificar si ya tenemos el contexto en caché
        if (messageContextStore.has(messageId)) {
            console.log(`Usando contexto en caché para mensaje: ${messageId}`);
            renderMessageContext(messageContextStore.get(messageId), contextContainer);
            return;
        }
        
        // Si no está en caché, obtenerlo de la API
        console.log(`Solicitando contexto a la API para mensaje: ${messageId}`);
        fetchMessageContext(messageId)
            .then(contextData => {
                if (!contextData || !contextData.context || contextData.context.length === 0) {
                    contextContainer.innerHTML = '<div class="no-context">No hay fuentes disponibles para este mensaje.</div>';
                    console.warn(`No se encontró contexto para el mensaje: ${messageId}`);
                    return;
                }
                
                // Almacenar el contexto en caché
                messageContextStore.set(messageId, contextData);
                
                // Renderizar el contexto
                renderMessageContext(contextData, contextContainer);
            })
            .catch(error => {
                console.error(`Error al obtener contexto para mensaje ${messageId}:`, error);
                contextContainer.innerHTML = '<div class="context-error">Error al cargar las fuentes. Intente nuevamente.</div>';
            });
    }

    // Función para renderizar el contexto del mensaje
    function renderMessageContext(contextData, container) {
        console.log('Renderizando contexto:', contextData);
        
        if (!contextData || !contextData.context || contextData.context.length === 0) {
            container.innerHTML = '<div class="no-context">No hay fuentes disponibles.</div>';
            return;
        }
        
        // Preparar el HTML para el contexto
        let contextHTML = `
            <div class="context-header">
                <h4>Fuentes utilizadas (${contextData.context.length})</h4>
                <p class="context-query">Consulta: "${contextData.query || 'No disponible'}"</p>
            </div>
            <div class="context-items">
        `;
        
        // Ordenar el contexto por relevancia
        const sortedContext = [...contextData.context].sort((a, b) => 
            (b.relevance || 0) - (a.relevance || 0)
        );
        
        // Añadir cada fragmento de contexto
        sortedContext.forEach((item, index) => {
            const relevancePercentage = Math.round((item.relevance || 0) * 100) / 100;
            const relevanceClass = 
                relevancePercentage > 80 ? 'high-relevance' : 
                relevancePercentage > 50 ? 'medium-relevance' : 'low-relevance';
            
            contextHTML += `
                <div class="context-item ${relevanceClass}">
                    <div class="context-item-header">
                        <span class="context-item-title">${item.header || item.title || 'Documento sin título'}</span>
                        <span class="context-item-relevance" title="Relevancia: ${relevancePercentage}%">
                            ${relevancePercentage}%
                        </span>
                    </div>
                    <div class="context-item-content">
                        ${item.content || 'No hay contenido disponible'}
                    </div>
                    <div class="context-item-meta">
                        ${item.source ? `<span class="context-item-source" title="${item.source}">Fuente: ${truncateText(item.source, 30)}</span>` : ''}
                        ${item.page ? `<span class="context-item-page">Página: ${item.page}</span>` : ''}
                    </div>
                </div>
            `;
        });
        
        contextHTML += `</div>`;
        
        // Actualizar el contenedor
        container.innerHTML = contextHTML;
    }

    // Función auxiliar para truncar texto largo
    function truncateText(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    // Inicialización
    function init() {
        console.log('Iniciando inicialización de la aplicación...');
        try {
            // Inicializar configuraciones
            initSettings();
            
            // Aplicar tema
            setTheme(settings.theme || 'dark');
            if (themeSelect) themeSelect.value = settings.theme || 'dark';
            if (modalThemeSelect) modalThemeSelect.value = settings.theme || 'dark';
            
            // Asegurar que la interfaz de chat está oculta inicialmente
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.style.display = 'none';
                console.log('Chat container oculto');
            } else {
                console.error('Error: No se encontró el elemento chat-container');
            }
            
            const inputArea = document.querySelector('.input-area');
            if (inputArea) {
                inputArea.style.display = 'none';
                console.log('Input area oculta');
            } else {
                console.error('Error: No se encontró el elemento input-area');
            }
            
            // Mostrar pantalla de bienvenida
            if (welcomeScreen) {
                welcomeScreen.style.display = 'flex';
                console.log('Welcome screen mostrada');
            } else {
                console.error('Error: No se encontró el elemento welcome-screen');
            }
            
            // Deshabilitar controles hasta que se seleccione una base de datos
            if (chatInput) chatInput.disabled = true;
            if (sendButton) sendButton.disabled = true;
            if (clearButton) clearButton.disabled = true;
            if (exportChatBtn) exportChatBtn.disabled = true;
            
            // Primero cargar la lista de bases de datos
            console.log('Iniciando carga de bases de datos...');
            loadDatabases();
            
            // Configurar vista móvil
            setupMobileView();
            
            console.log('Aplicación inicializada correctamente');
        } catch (err) {
            console.error('Error durante la inicialización de la aplicación:', err);
        }
    }

    // Llamar a la función de inicialización
    // document.addEventListener('DOMContentLoaded', init);
    // Ejecutamos init directamente, ya que ya estamos dentro de un DOMContentLoaded
    init();
    
    // Eventos adicionales para mejorar la responsividad
    window.addEventListener('resize', function() {
        setupMobileView();
        adjustTextareaHeight(chatInput);
    });

    // Detectar cambios en la orientación del dispositivo
    window.addEventListener('orientationchange', function() {
        setTimeout(setupMobileView, 100);
        setTimeout(function() {
            adjustTextareaHeight(chatInput);
            scrollToBottom();
        }, 200);
    });

    // Escuchar cambios en la configuración
    document.addEventListener('settingsChanged', function(e) {
        if (e.detail && e.detail.reduceMotion !== undefined) {
            if (e.detail.reduceMotion) {
                document.body.classList.add('reduce-motion');
            } else {
                document.body.classList.remove('reduce-motion');
            }
        }
    });

    // Evento para selector de tema del modal (sincroniza con el selector principal)
    modalThemeSelect.addEventListener('change', () => {
        themeSelect.value = modalThemeSelect.value;
        setTheme(modalThemeSelect.value);
    });
    
    // Al abrir el modal, sincronizar el selector de tema
    settingsButton.addEventListener('click', () => {
        modalThemeSelect.value = themeSelect.value;
        settingsModal.classList.add('active');
    });
    
    inputSettingsButton.addEventListener('click', () => {
        modalThemeSelect.value = themeSelect.value;
        settingsModal.classList.add('active');
    });
});
