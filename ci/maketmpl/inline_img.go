package maketmpl

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func inlineImgSrc(srcDir, src string) (string, error) {
	// We do not inline full URLs.
	if strings.HasPrefix(src, "http://") || strings.HasPrefix(src, "https://") {
		return src, nil
	}

	var dataType string
	switch {
	case strings.HasSuffix(src, ".png"):
		dataType = "image/png"
	case strings.HasSuffix(src, ".jpg") || strings.HasSuffix(src, ".jpeg"):
		dataType = "image/jpeg"
	case strings.HasSuffix(src, ".gif"):
		dataType = "image/gif"
	default:
		return "", fmt.Errorf("unsupported image type: %s", src)
	}

	fp := src
	if srcDir != "" {
		fp = filepath.Join(srcDir, fp)
	}
	imgData, err := os.ReadFile(fp)
	if err != nil {
		return "", fmt.Errorf("read image file: %w", err)
	}

	encoded := base64.StdEncoding.EncodeToString(imgData)
	return fmt.Sprintf("data:%s;base64,%s", dataType, encoded), nil
}
