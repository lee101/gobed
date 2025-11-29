package gobed

import (
	"io"
	"log"
	"os"
	"sync"
	"sync/atomic"
)

var (
	debugLogger    = log.New(os.Stderr, "gobed: ", log.LstdFlags)
	debugLoggerMu  sync.Mutex
	debugLoggingOn atomic.Bool
)

func init() {
	if _, ok := os.LookupEnv("GOBED_DEBUG"); ok {
		debugLoggingOn.Store(true)
	}
}

// EnableDebugLogging globally enables debug logging output.
func EnableDebugLogging() {
	debugLoggingOn.Store(true)
}

// DisableDebugLogging globally disables debug logging output.
func DisableDebugLogging() {
	debugLoggingOn.Store(false)
}

// DebugLoggingEnabled returns true when debug logging is currently enabled.
func DebugLoggingEnabled() bool {
	return debugLoggingOn.Load()
}

// SetDebugOutput overrides the writer used for debug output.
func SetDebugOutput(w io.Writer) {
	if w == nil {
		w = io.Discard
	}
	debugLoggerMu.Lock()
	debugLogger.SetOutput(w)
	debugLoggerMu.Unlock()
}

// Debugf emits a formatted debug message when debug logging is enabled.
func Debugf(format string, args ...interface{}) {
	if debugLoggingOn.Load() {
		debugLogger.Printf(format, args...)
	}
}

// Debugln emits a debug message with default formatting when enabled.
func Debugln(args ...interface{}) {
	if debugLoggingOn.Load() {
		debugLogger.Println(args...)
	}
}
