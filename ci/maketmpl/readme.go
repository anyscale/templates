package maketmpl

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

type readmeFile struct {
	notebookFile string

	md   []byte
	imgs []*mdImage
}

func (f *readmeFile) writeInto(w io.Writer, imgOpts *writeImgOptions) error {
	// Copies the markdown file into writer w, but rewrites all images
	// in the markdown file following options in imgOpts

	// cursor is the point where things have copied. It is an offset in the
	// source markdown file, which is f.md
	cursor := 0

	for i, img := range f.imgs {
		// Copy any part that is before the next image first.
		// img.start is the starting point of the markdown image.
		if cursor < img.start {
			if _, err := w.Write(f.md[cursor:img.start]); err != nil {
				return fmt.Errorf("write markdown: %w", err)
			}
		}

		// Write out the image.
		if err := img.writeInto(w, imgOpts); err != nil {
			return fmt.Errorf("write image %d: %w", i, err)
		}

		// Forward the read cursor to the end of the image.
		cursor = img.end
	}

	// Copy the rest.
	if cursor < len(f.md) {
		if _, err := w.Write(f.md[cursor:]); err != nil {
			return fmt.Errorf("write markdown: %w", err)
		}
	}

	return nil
}

func (f *readmeFile) writeIntoFile(p string, imgOpts *writeImgOptions) error {
	out, err := os.Create(p)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer out.Close()

	if err := f.writeInto(out, imgOpts); err != nil {
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

func (f *readmeFile) writeGitHubMD(path string) error {
	// GitHub flavored markdown does not support inline images
	// and forbids having inlined styles.
	imgOpts := &writeImgOptions{
		inlineSrc:   false,
		sizeInStyle: false,
	}
	return f.writeIntoFile(path, imgOpts)
}

func (f *readmeFile) writeReleaseMD(path, baseDir string) error {
	// This is for rendering in doc pages and other Web sites.
	// and we want the file to be reliable, consistent and self-contained.
	imgOpts := &writeImgOptions{
		inlineSrc:    true,
		inlineSrcDir: baseDir,
		sizeInStyle:  true,
	}
	return f.writeIntoFile(path, imgOpts)
}

func readReadmeFile(path string) (*readmeFile, error) {
	md, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	imgs, err := parseMdImages(md)
	if err != nil {
		return nil, fmt.Errorf("parse images: %w", err)
	}

	return &readmeFile{
		md:   md,
		imgs: imgs,
	}, nil
}

func readmeFromNotebook(f string) (*readmeFile, error) {
	stat, err := os.Stat(f)
	if err != nil {
		return nil, fmt.Errorf("stat notebook file: %w", err)
	}
	if stat.IsDir() {
		return nil, fmt.Errorf("notebook is a directory")
	}

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
	log.Println(cmd.Args)

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("convert notebook: %w", err)
	}

	outputFile := filepath.Join(tmpDir, "README.md")

	readme, err := readReadmeFile(outputFile)
	if err != nil {
		return nil, err
	}
	readme.notebookFile = f

	return readme, nil
}
