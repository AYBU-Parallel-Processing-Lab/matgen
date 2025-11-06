#include "matgen/utils/log.h"

#include <stdarg.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

// =============================================================================
// Global State
// =============================================================================

static struct {
  matgen_log_level_t level;
  FILE* stream;
  bool timestamps;
  bool color;
  bool color_initialized;
} g_log_config = {.level = MATGEN_LOG_LEVEL_INFO,
                  .stream = NULL,  // Will default to stderr on first use
                  .timestamps = true,
                  .color = false,  // Disabled by default, will auto-detect
                  .color_initialized = false};

// =============================================================================
// ANSI Color Codes
// =============================================================================

#define COLOR_RESET "\033[0m"
#define COLOR_TRACE "\033[37m"    // White
#define COLOR_DEBUG "\033[36m"    // Cyan
#define COLOR_INFO "\033[32m"     // Green
#define COLOR_WARN "\033[33m"     // Yellow
#define COLOR_ERROR "\033[31m"    // Red
#define COLOR_FATAL "\033[35;1m"  // Bold Magenta

// =============================================================================
// Windows VT Support
// =============================================================================

#ifdef _WIN32

/**
 * @brief Enable Windows Virtual Terminal processing for ANSI escape codes
 *
 * This enables colored output on Windows 10 (build 1511+) and later.
 *
 * @param stream The file stream to enable VT processing for
 * @return true if VT mode was enabled, false otherwise
 */
static bool enable_windows_vt_mode(FILE* stream) {
  if (!stream) {
    return false;
  }

  // Get the file descriptor
  int fd = fileno(stream);
  if (fd < 0) {
    return false;
  }

  // Check if it's a terminal
  if (!isatty(fd)) {
    return false;
  }

  // Get the Windows console handle
  HANDLE hConsole;
  if (fd == fileno(stdout)) {
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  } else if (fd == fileno(stderr)) {
    hConsole = GetStdHandle(STD_ERROR_HANDLE);
  } else {
    return false;
  }

  if (hConsole == INVALID_HANDLE_VALUE) {
    return false;
  }

  // Get current console mode
  DWORD dwMode = 0;
  if (!GetConsoleMode(hConsole, &dwMode)) {
    return false;
  }

  // Enable Virtual Terminal Processing
  dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
  if (!SetConsoleMode(hConsole, dwMode)) {
    return false;
  }

  return true;
}

#else

/**
 * @brief Check if the stream supports ANSI colors (Unix/Linux/macOS)
 *
 * @param stream The file stream to check
 * @return true if ANSI colors are supported, false otherwise
 */
static bool supports_ansi_colors(FILE* stream) {
  if (!stream) {
    return false;
  }

  int fd = fileno(stream);
  if (fd < 0) {
    return false;
  }

  // Check if it's a terminal
  if (!isatty(fd)) {
    return false;
  }

  // Check TERM environment variable
  const char* term = getenv("TERM");
  if (!term || strcmp(term, "dumb") == 0) {
    return false;
  }

  return true;
}

#endif

/**
 * @brief Initialize color support (auto-detect or force enable)
 *
 * @param stream The stream to initialize colors for
 */
static void init_color_support(FILE* stream) {
  if (g_log_config.color_initialized) {
    return;
  }

  bool supported = false;

#ifdef _WIN32
  supported = enable_windows_vt_mode(stream);
#else
  supported = supports_ansi_colors(stream);
#endif

  // Auto-enable if supported
  if (supported && !g_log_config.color_initialized) {
    g_log_config.color = true;
  }

  g_log_config.color_initialized = true;
}

// =============================================================================
// Configuration Functions
// =============================================================================

void matgen_log_set_level(matgen_log_level_t level) {
  g_log_config.level = level;
}

matgen_log_level_t matgen_log_get_level(void) { return g_log_config.level; }

void matgen_log_set_stream(FILE* stream) {
  g_log_config.stream = stream;
  g_log_config.color_initialized =
      false;  // Re-check color support for new stream
}

void matgen_log_set_timestamps(bool enabled) {
  g_log_config.timestamps = enabled;
}

void matgen_log_set_color(bool enabled) {
  g_log_config.color = enabled;
  g_log_config.color_initialized =
      true;  // User explicitly set color preference

  // If enabling, try to initialize VT mode
  if (enabled) {
    FILE* stream = g_log_config.stream ? g_log_config.stream : stderr;
#ifdef _WIN32
    enable_windows_vt_mode(stream);
#endif
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

static const char* level_string(matgen_log_level_t level) {
  switch (level) {
    case MATGEN_LOG_LEVEL_TRACE:
      return "TRACE";
    case MATGEN_LOG_LEVEL_DEBUG:
      return "DEBUG";
    case MATGEN_LOG_LEVEL_INFO:
      return "INFO ";
    case MATGEN_LOG_LEVEL_WARN:
      return "WARN ";
    case MATGEN_LOG_LEVEL_ERROR:
      return "ERROR";
    case MATGEN_LOG_LEVEL_FATAL:
      return "FATAL";
    default:
      return "?????";
  }
}

static const char* level_color(matgen_log_level_t level) {
  switch (level) {
    case MATGEN_LOG_LEVEL_TRACE:
      return COLOR_TRACE;
    case MATGEN_LOG_LEVEL_DEBUG:
      return COLOR_DEBUG;
    case MATGEN_LOG_LEVEL_INFO:
      return COLOR_INFO;
    case MATGEN_LOG_LEVEL_WARN:
      return COLOR_WARN;
    case MATGEN_LOG_LEVEL_ERROR:
      return COLOR_ERROR;
    case MATGEN_LOG_LEVEL_FATAL:
      return COLOR_FATAL;
    default:
      return COLOR_RESET;
  }
}

static void get_timestamp(char* buffer, size_t size) {
  time_t now = time(NULL);
  struct tm* tm_info = localtime(&now);
  strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

static const char* basename_only(const char* path) {
  const char* base = strrchr(path, '/');
  if (!base) {
    base = strrchr(path, '\\');
  }
  return base ? base + 1 : path;
}

// =============================================================================
// Main Logging Function
// =============================================================================

void matgen_log(matgen_log_level_t level, const char* file, int line,
                const char* func, const char* fmt, ...) {
  // Check if we should log this message
  if (level < g_log_config.level) {
    return;
  }

  // Default to stderr if no stream set
  FILE* out = g_log_config.stream ? g_log_config.stream : stderr;

  // Initialize color support on first use
  if (!g_log_config.color_initialized) {
    init_color_support(out);
  }

  // Build the log message
  char timestamp[32] = "";
  if (g_log_config.timestamps) {
    get_timestamp(timestamp, sizeof(timestamp));
  }

  // Print header with optional color
  if (g_log_config.color) {
    fprintf(out, "%s", level_color(level));
  }

  if (g_log_config.timestamps) {
    fprintf(out, "[%s] ", timestamp);
  }

  fprintf(out, "[%s] ", level_string(level));

  // For debug/trace, show file location
  if (level <= MATGEN_LOG_LEVEL_DEBUG) {
    fprintf(out, "[%s:%d %s] ", basename_only(file), line, func);
  }

  // Print the actual message
  va_list args;
  va_start(args, fmt);
  vfprintf(out, fmt, args);
  va_end(args);

  // Reset color and add newline
  if (g_log_config.color) {
    fprintf(out, "%s", COLOR_RESET);
  }

  fprintf(out, "\n");
  fflush(out);
}
