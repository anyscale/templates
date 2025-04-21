package maketmpl

import (
	"os"
	"path/filepath"
	"testing"
)

func checkBuiltTemplate(t *testing.T, tmp, tmplName string) {
	t.Helper()

	// Check that critical files are generated.
	for _, f := range []string{
		buildDotZip,
		rayAppDotJSON,
		readmeDocMD,
		readmeGitHubMD,
	} {
		stat, err := os.Stat(filepath.Join(tmp, tmplName, f))
		if err != nil {
			t.Errorf("os.Stat(%q): %v", f, err)
			continue
		}

		if stat.IsDir() {
			t.Errorf("%q is a directory", f)
		}
		if stat.Size() == 0 {
			t.Errorf("%q is empty", f)
		}
	}
}

func TestBuildAll(t *testing.T) {
	tmp := t.TempDir()

	if err := BuildAll("testdata/BUILD.yaml", "testdata", tmp); err != nil {
		t.Fatal(err)
	}

	for _, tmpl := range []string{"reefy-ray", "fishy-ray"} {
		checkBuiltTemplate(t, tmp, tmpl)
	}
}

func TestBuild(t *testing.T) {
	tmp := t.TempDir()

	if err := Build(
		"testdata/BUILD.yaml", "reefy-ray", "testdata", tmp,
	); err != nil {
		t.Fatal(err)
	}

	checkBuiltTemplate(t, tmp, "reefy-ray")
}

func TestBuild_notFound(t *testing.T) {
	tmp := t.TempDir()

	if err := Build(
		"testdata/BUILD.yaml", "not-found", "testdata", tmp,
	); err != errNoTemplateBuilt {
		t.Fatalf("want error %q, got %q", errNoTemplateBuilt, err)
	}
}
