package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Same column order as the notebook's outputs/customer_recommendations.csv.
const header = "customer_index,churn_probability,retention_uplift,expected_profit,segment,recommended_intervention,actual_treatment,actual_outcome\n"

func writeCSV(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "customer_recommendations.csv")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadCSVUsesRetentionUpliftColumn(t *testing.T) {
	path := writeCSV(t, header+
		"106,0.147,0.0167,-13.2,Unaffected,No_Action,1,0\n"+
		"521,0.128,-0.0300,-88.0,Sleeping_Dog,No_Action,0,1\n")

	got, err := readCSV(path)
	if err != nil {
		t.Fatal(err)
	}
	want := []ChurnData{{"106", 0.0167}, {"521", -0.0300}}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestReadCSVReportsBadInput(t *testing.T) {
	cases := map[string]string{
		"missing file":   filepath.Join(t.TempDir(), "absent.csv"),
		"missing column": writeCSV(t, "customer_index,churn_probability\n1,0.5\n"),
		"bad number":     writeCSV(t, header+"1,0.5,abc,0,Unaffected,No_Action,1,0\n"),
		"ragged row":     writeCSV(t, header+"1,0.5\n"),
		"empty file":     writeCSV(t, ""),
	}
	for name, path := range cases {
		if _, err := readCSV(path); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestTopByUpliftSortsAndTruncates(t *testing.T) {
	data := []ChurnData{{"a", 0.01}, {"b", 0.05}, {"c", -0.02}, {"d", 0.03}}

	got := topByUplift(data, 2)

	if fmt.Sprint(got) != fmt.Sprint([]ChurnData{{"b", 0.05}, {"d", 0.03}}) {
		t.Fatalf("got %v", got)
	}
	if data[0].CustomerID != "a" {
		t.Fatal("input slice was reordered")
	}
	if len(topByUplift(data, 10)) != 4 {
		t.Fatal("short input should be returned whole")
	}
}

func TestDashboardRendersTopCustomers(t *testing.T) {
	path := writeCSV(t, header+
		"7,0.2,0.0100,-24,Unaffected,No_Action,1,0\n"+
		"42,0.3,0.0900,104,Persuadable,Retention_Offer,1,0\n")
	rec := httptest.NewRecorder()

	dashboardHandler(path)(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, body)
	}
	for _, want := range []string{"Customer Uplift Analysis", "Top 2 customers by retention uplift", `"42"`} {
		if !strings.Contains(body, want) {
			t.Errorf("response does not contain %q", want)
		}
	}
	if strings.Index(body, `"42"`) > strings.Index(body, `"7"`) {
		t.Error("highest-uplift customer should come first")
	}
}

func TestDashboardExplainsMissingOutputs(t *testing.T) {
	rec := httptest.NewRecorder()

	dashboardHandler(filepath.Join(t.TempDir(), "absent.csv"))(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status %d, want 500", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "Run Churn_Uplift_COMPLETE.ipynb first") {
		t.Errorf("unhelpful error: %s", rec.Body.String())
	}
}
