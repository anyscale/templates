package maketmpl

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func readFileInZip(f *zip.File) ([]byte, error) {
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	buf := new(bytes.Buffer)
	if _, err := io.Copy(buf, rc); err != nil {
		return nil, err
	}
	if err := rc.Close(); err != nil {
		return nil, fmt.Errorf("close: %w", err)
	}
	return buf.Bytes(), nil
}

func fileNameList(files map[string][]byte) []string {
	var names []string
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func TestBuilder(t *testing.T) {
	checkJupyterOrSkipOnLocal(t)

	tmp := t.TempDir()

	tmpl := &Template{
		Name:  "reefy-ray",
		Dir:   "reefy-ray",
		Title: "Ray lives in the REEf",
		ClusterEnv: map[string]any{
			"build_id": "anyscaleray2340-py311",
		},
		ComputeConfig: map[string]string{
			"AWS": "testdata/configs/aws.yaml",
			"GCP": "testdata/configs/gcp.yaml",
		},
	}

	b := newBuilder(tmpl, "testdata")

	if err := b.build(tmp); err != nil {
		t.Fatal("build: ", err)
	}

	// Check the output files.
	buildZip := filepath.Join(tmp, buildDotZip)

	z, err := zip.OpenReader(buildZip)
	if err != nil {
		t.Fatalf("open built zip: %v", err)
	}
	defer z.Close()

	gotFiles := make(map[string][]byte)

	for _, f := range z.File {
		content, err := readFileInZip(f)
		if err != nil {
			t.Fatalf("read file %q: %v", f.Name, err)
		}
		gotFiles[f.Name] = content
	}

	gotFileNames := fileNameList(gotFiles)

	srcReadme, err := os.ReadFile("testdata/reefy-ray/README.ipynb")
	if err != nil {
		t.Fatal("read src readme: ", err)
	}
	srcPng, err := os.ReadFile("testdata/reefy-ray/a.png")
	if err != nil {
		t.Fatal("read src png: ", err)
	}

	wantFiles := map[string][]byte{
		"README.ipynb": srcReadme,
		"a.png":        srcPng,
	}
	wantFileNames := fileNameList(wantFiles)

	if !reflect.DeepEqual(gotFileNames, wantFileNames) {
		t.Fatalf("got files %v, want %v", gotFileNames, wantFileNames)
	}

	for name, want := range wantFiles {
		got := gotFiles[name]
		if !bytes.Equal(got, want) {
			t.Errorf("file %q: got %q, want %q", name, got, want)
		}
	}
}
