package maketmpl

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

type readmeFile struct {
	notebookFile string

	md   []byte
	imgs []*mdImage
}

func (f *readmeFile) writeInto(
	w io.Writer, baseDir string, inlineImg bool,
) error {
	cursor := 0
	for i, img := range f.imgs {
		if cursor < img.start {
			if _, err := w.Write(f.md[cursor:img.start]); err != nil {
				return fmt.Errorf("write markdown: %w", err)
			}
		}

		if err := img.writeInto(w, baseDir, inlineImg); err != nil {
			return fmt.Errorf("write image %d: %w", i, err)
		}

		cursor = img.end
	}

	if cursor < len(f.md) {
		if _, err := w.Write(f.md[cursor:]); err != nil {
			return fmt.Errorf("write markdown: %w", err)
		}
	}

	return nil
}

func (f *readmeFile) writeIntoFile(path, baseDir string, inlineImg bool) error {
	out, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer out.Close()

	if err := f.writeInto(out, baseDir, inlineImg); err != nil {
		return err
	}

	if err := out.Sync(); err != nil {
		return fmt.Errorf("sync output file: %w", err)
	}

	if err := out.Close(); err != nil {
		return fmt.Errorf("close output file: %w", err)
	}

	return nil
}

func (f *readmeFile) writeGitHubMD(path, baseDir string) error {
	return f.writeIntoFile(path, baseDir, false /* inlineImg */)
}

func (f *readmeFile) writeReleaseMD(path, baseDir string) error {
	return f.writeIntoFile(path, baseDir, true /* inlineImg */)
}

func buildReadme(f string) (*readmeFile, error) {
	tmpDir, err := os.MkdirTemp("", "maketmpl_*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	cmd := exec.Command(
		"jupyter", "nbconvert", "--to", "markdown",
		f, "--output", "README",
		"--output-dir", tmpDir,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("convert notebook: %w", err)
	}

	outputFile := filepath.Join(tmpDir, "README.md")
	md, err := os.ReadFile(outputFile)
	if err != nil {
		return nil, fmt.Errorf("read output markdown file: %w", err)
	}

	imgs, err := parseMdImages(md)
	if err != nil {
		return nil, fmt.Errorf("parse images: %w", err)
	}

	return &readmeFile{
		notebookFile: f,

		md:   md,
		imgs: imgs,
	}, nil
}