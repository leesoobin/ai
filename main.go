package main

import (
	"context"
	"fmt"
	"log"

	"github.com/ollama/ollama/api"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

const (
	ollamaModel     = "llama3.2"
	weaviateHost    = "localhost:8080"
	className       = "Sbb"
	vectorDimension = 3072 // Ollama의 임베딩 차원에 맞게 조정
)

func main() {
	ctx := context.Background()

	// Ollama 클라이언트 초기화
	ollamaClient, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatalf("Ollama 클라이언트 생성 실패: %v", err)
	}

	// Weaviate 클라이언트 초기화
	weaviateClient, err := weaviate.NewClient(weaviate.Config{
		Host:   weaviateHost,
		Scheme: "http", // 프로토콜 설정
	})
	if err != nil {
		log.Fatalf("Weaviate 연결 실패: %v", err)
	}

	클래스 생성 (이미 존재하지 않는 경우)
	createClass(ctx, weaviateClient)

	문서 인덱싱
	documents := []string{
		"sb.lee의 서버접근권한 정보 서버:ap-sec-smg01,계정:hiware",
		"sb.lee의 서버접근권한 정보 서버:ap-sec-smg01,계정:nkia",
		"aj.hyun의 서버접근권한 정보 서버:ap-sec-smg01,계정:sre",
	}
	indexDocuments(ctx, ollamaClient, weaviateClient, documents)

	// 사용자 쿼리 처리
	query := "sb.lee의 서버접근권한 알려줘"
	answer := processQuery(ctx, ollamaClient, weaviateClient, query)
	fmt.Printf("질문: %s\n답변: %s\n", query, answer)
}

func createClass(ctx context.Context, client *weaviate.Client) {
	classObj := &models.Class{
		Class: className,
		Properties: []*models.Property{
			{
				Name:     "text",
				DataType: []string{"text"},
			},
		},
		VectorIndexType: "hnsw",
	}

	err := client.Schema().ClassCreator().WithClass(classObj).Do(ctx)
	if err != nil {
		log.Printf("클래스 생성 실패 (이미 존재할 수 있음): %v", err)
	}
}

func indexDocuments(ctx context.Context, ollamaClient *api.Client, weaviateClient *weaviate.Client, documents []string) {
	for _, doc := range documents {
		// 임베딩 생성
		resp, err := ollamaClient.Embeddings(ctx, &api.EmbeddingRequest{
			Model:  ollamaModel,
			Prompt: doc,
		})
		if err != nil {
			log.Printf("문서 임베딩 실패: %v", err)
			continue
		}

		// Weaviate에 벡터 삽입
		_, err = weaviateClient.Data().Creator().
			WithClassName(className).
			WithProperties(map[string]interface{}{
				"text": doc,
			}).
			WithVector(convertToFloat32(resp.Embedding)).
			Do(ctx)

		if err != nil {
			log.Printf("벡터 삽입 실패: %v", err)
		}
	}
}

func processQuery(ctx context.Context, ollamaClient *api.Client, weaviateClient *weaviate.Client, query string) string {
	// 쿼리 임베딩
	resp, err := ollamaClient.Embeddings(ctx, &api.EmbeddingRequest{
		Model:  ollamaModel,
		Prompt: query,
	})
	if err != nil {
		log.Printf("쿼리 임베딩 실패: %v", err)
		return "죄송합니다. 쿼리 처리 중 오류가 발생했습니다."
	}

	// 유사한 문서 검색
	nearVector := weaviateClient.GraphQL().NearVectorArgBuilder().
		WithVector(convertToFloat32(resp.Embedding))

	result, err := weaviateClient.GraphQL().Get().
		WithClassName(className).
		WithFields(graphql.Field{Name: "text"}).
		WithNearVector(nearVector).
		WithLimit(3).
		Do(ctx)

	if err != nil {
		log.Printf("유사 문서 검색 실패: %v", err)
		return "죄송합니다. 관련 정보를 찾는 중 오류가 발생했습니다."
	}

	// 컨텍스트 구성
	context := "다음 정보를 바탕으로 질문에 답하세요:\n"
	if data, ok := result.Data["Get"].(map[string]interface{}); ok {
		if documents, ok := data[className].([]interface{}); ok {
			for _, doc := range documents {
				if item, ok := doc.(map[string]interface{}); ok {
					if text, ok := item["text"].(string); ok {
						context += text + "\n"
					}
				}
			}
		}
	}
	fmt.Println("Context:", context)

	// LLM에 질의
	var answer string
	err = ollamaClient.Generate(ctx, &api.GenerateRequest{
		Model:  ollamaModel,
		Prompt: context + "\n질문: " + query + "\n답변:",
	}, func(response api.GenerateResponse) error {
		answer += response.Response
		return nil
	})
	if err != nil {
		log.Printf("LLM 응답 생성 실패: %v", err)
		return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
	}

	return answer
}

func convertToFloat32(f64 []float64) []float32 {
	f32 := make([]float32, len(f64))
	for i, v := range f64 {
		f32[i] = float32(v)
	}
	return f32
}
