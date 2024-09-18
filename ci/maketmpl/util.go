package maketmpl

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
)

func checkIsDir(path string) error {
	stat, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !stat.IsDir() {
		return fmt.Errorf("%s is not a directory", path)
	}
	return nil
}

func addFileToZip(z *zip.Writer, file, path string) error {
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("open file %q: %w", file, err)
	}
	defer f.Close()

	w, err := z.Create(path)
	if err != nil {
		return fmt.Errorf("create file in zip: %w", err)
	}
	if _, err := io.Copy(w, f); err != nil {
		return fmt.Errorf("copy file to zip: %w", err)
	}
	return nil
}
