const { MongoClient } = require('mongodb');
const Groq = require('groq-sdk');
require('dotenv').config();

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});

const mongoUri = process.env.MONGODB_URI || 'mongodb://localhost:27017';
const dbName = process.env.DB_NAME;
const collectionName = 'customer_faq';

const BATCH_SIZE = parseInt(process.env.BATCH_SIZE) || 5;
const DELAY_BETWEEN_BATCHES = parseInt(process.env.BATCH_DELAY) || 2000; // Delay between batches in ms

const evaluationCriteria = [
    'completeness', // How complete the answer is
    'accuracy', // How accurate the information appears
    'clarity', // How clear and understandable the answer is
    'usefulness' // How useful the answer would be to the user
];

async function fetchFAQsFromMongo() {
    const client = new MongoClient(mongoUri);

    try {
        await client.connect();
        console.log('Connected to MongoDB');

        const db = client.db(dbName);
        const collection = db.collection(collectionName);

        const faqs = await collection.find({}).toArray();
        console.log(`Fetched ${faqs.length} FAQ documents`);

        return faqs;
    } catch (error) {
        console.error('Error fetching data from MongoDB:', error);
        throw error;
    } finally {
        await client.close();
    }
}

async function evaluateRelevance(question, answer, category = null) {

    const prompt = `
    You are an expert FAQ quality evaluator. Please evaluate the following answer against the given question using these specific criteria:

    **Evaluation Criteria:**
    ${evaluationCriteria.map(criteria => `- ${criteria}`).join('\n')}

    **Question:** "${question}"
    **Answer:** "${answer}"
    ${category ? `**Category:** ${category}` : ''}

    **Detailed Scoring Guidelines:**

    **Completeness (1-10):**
    - 10: Provides comprehensive step-by-step instructions, covers all necessary details, anticipates follow-up questions
    - 7-9: Includes most necessary steps but may miss 1-2 minor details
    - 4-6: Provides basic information but lacks important steps or context
    - 1-3: Vague, incomplete, or missing critical information (e.g., "You can do X from the app" without explaining how)

    **Accuracy (1-10):**
    - 10: All information is factually correct and up-to-date
    - 5: Mostly correct with minor inaccuracies
    - 1: Contains significant errors or misleading information

    **Clarity (1-10):**
    - 10: Uses clear, simple language that any user can understand, well-structured
    - 5: Generally clear but some confusing elements
    - 1: Confusing, poorly written, or uses excessive jargon

    **Usefulness/Actionability (1-10):**
    - 10: User can immediately act on the answer and achieve their goal
    - 7-9: User can mostly act on the answer with minimal additional research
    - 4-6: User gets general direction but needs additional information
    - 1-3: Answer doesn't help user achieve their goal (red flag for answers like "You can change X from the app" without actual steps)

    **Special Instructions:**
    - Heavily penalize answers that are technically correct but lack actionable steps
    - Look for answers that sound helpful but don't actually help users complete tasks
    - Reward answers that include specific UI elements, button names, or navigation paths
    - Consider whether a typical user could successfully complete the task using only the provided answer

    Rate each criterion on a scale of 1-10. Provide your response in this exact JSON format:

    {
    "completeness": score, 
    "accuracy": score,
    "clarity": score,
    "usefulness": score,
    "overall_score": calculated_average,
    "reasoning": "Specific explanation of why scores were given, especially highlighting any missing steps or vague language",
    "improvement_suggestions": "Concrete suggestions for making the answer more helpful"
    }
    `;
    try {
        const completion = await groq.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content: "You are an expert evaluator of FAQ content. Provide objective, numerical assessments of answer relevance."
                },
                {
                    role: "user",
                    content: prompt
                }
            ],
            model: "llama-3.1-8b-instant",
            temperature: 0.3,
            max_tokens: 500
        });

        const response = completion.choices[0]?.message?.content;

        if (!response) {
            throw new Error('Empty response from LLM');
        }

        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        } else {
            throw new Error('Invalid response format from LLM');
        }
    } catch (error) {
        console.error('Error evaluating relevance:', error);
        return {
            completeness: 0,
            accuracy: 0,
            clarity: 0,
            usefulness: 0,
            overall_score: 0,
            reasoning: "Error occurred during evaluation"
        };
    }
}

async function processSingleFAQ(faq, index, total) {
    try {
        console.log(`Processing FAQ ${index + 1}/${total}: ${faq._id}`);

        const evaluation = await evaluateRelevance(
            faq.question,
            faq.answer,
            `${faq.l1_category} - ${faq.l2_category}`
        );

        return {
            faq_id: faq._id,
            question: faq.question,
            answer: faq.answer,
            category: {
                l1: faq.l1_category,
                l2: faq.l2_category
            },
            language: faq.language,
            evaluation: evaluation,
            processed_at: new Date().toISOString(),
            success: true
        };

    } catch (error) {
        console.error(`Error processing FAQ ${faq._id}:`, error);
        return {
            faq_id: faq._id,
            question: faq.question,
            answer: faq.answer,
            category: {
                l1: faq.l1_category,
                l2: faq.l2_category
            },
            language: faq.language,
            evaluation: null,
            error: error.message,
            processed_at: new Date().toISOString(),
            success: false
        };
    }
}

function createBatches(array, batchSize) {
    const batches = [];
    for (let i = 0; i < array.length; i += batchSize) {
        batches.push(array.slice(i, i + batchSize));
    }
    return batches;
}

// Optimized function using Promise.all with batching
async function processAllFAQsConcurrent() {
    try {
        console.log('Starting FAQ relevance evaluation with concurrent processing...');
        console.log(`Batch size: ${BATCH_SIZE}, Delay between batches: ${DELAY_BETWEEN_BATCHES}ms`);

        const faqs = await fetchFAQsFromMongo();

        if (faqs.length === 0) {
            console.log('No FAQ data found');
            return [];
        }

        const batches = createBatches(faqs, BATCH_SIZE);
        console.log(`Created ${batches.length} batches for processing`);

        const allResults = [];

        for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
            const batch = batches[batchIndex];
            const startTime = Date.now();

            console.log(`\nProcessing batch ${batchIndex + 1}/${batches.length} (${batch.length} FAQs)`);

            try {
                const batchPromises = batch.map((faq, index) =>
                    processSingleFAQ(faq, batchIndex * BATCH_SIZE + index, faqs.length)
                );

                const batchResults = await Promise.all(batchPromises);
                allResults.push(...batchResults);

                const endTime = Date.now();
                const batchDuration = endTime - startTime;

                console.log(`Batch ${batchIndex + 1} completed in ${batchDuration}ms`);
                console.log(`Successful: ${batchResults.filter(r => r.success).length}, Failed: ${batchResults.filter(r => !r.success).length}`);

                if (batchIndex < batches.length - 1) {
                    console.log(`Waiting ${DELAY_BETWEEN_BATCHES}ms before next batch...`);
                    await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES));
                }

            } catch (error) {
                console.error(`Error processing batch ${batchIndex + 1}:`, error);

                const errorResults = batch.map((faq, index) => ({
                    faq_id: faq._id,
                    question: faq.question,
                    answer: faq.answer,
                    evaluation: null,
                    error: `Batch processing failed: ${error.message}`,
                    processed_at: new Date().toISOString(),
                    success: false
                }));

                allResults.push(...errorResults);
            }
        }

        return allResults;
    } catch (error) {
        console.error('Error in concurrent processing:', error);
        throw error;
    }
}

async function processAllFAQsWithLimit(concurrencyLimit = BATCH_SIZE) {
    try {
        console.log(`Starting FAQ evaluation with concurrency limit: ${concurrencyLimit}`);

        const faqs = await fetchFAQsFromMongo();

        if (faqs.length === 0) {
            console.log('No FAQ data found');
            return [];
        }

        const results = [];
        const processing = [];

        for (let i = 0; i < faqs.length; i++) {
            const promise = processSingleFAQ(faqs[i], i, faqs.length)
                .then(result => {
                    results.push(result);
                    return result;
                });

            processing.push(promise);

            if (processing.length >= concurrencyLimit || i === faqs.length - 1) {
                await Promise.all(processing);
                processing.length = 0;

                if (i < faqs.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }
        }

        return results;
    } catch (error) {
        console.error('Error in limited concurrent processing:', error);
        throw error;
    }
}

async function saveResults(results, filename = 'faq_relevance_results.json') {
    const fs = require('fs').promises;

    try {
        await fs.writeFile(filename, JSON.stringify(results, null, 2));
        console.log(`Results saved to ${filename}`);

        // Also save a summary
        const summary = {
            total_faqs: results.length,
            successful_evaluations: results.filter(r => r.success).length,
            failed_evaluations: results.filter(r => !r.success).length,
            average_scores: calculateAverageScores(results),
            processing_time: new Date().toISOString(),
            batch_size: BATCH_SIZE,
            generated_at: new Date().toISOString()
        };

        await fs.writeFile('summary.json', JSON.stringify(summary, null, 2));
        console.log('Summary saved to summary.json');

    } catch (error) {
        console.error('Error saving results:', error);
    }
}

function calculateAverageScores(results) {
    const validResults = results.filter(r => r.success && r.evaluation !== null);

    if (validResults.length === 0) return null;

    const totals = evaluationCriteria.reduce((acc, criteria) => {
        acc[criteria] = 0;
        return acc;
    }, {});
    totals.overall_score = 0;

    validResults.forEach(result => {
        evaluationCriteria.forEach(criteria => {
            totals[criteria] += result.evaluation[criteria] || 0;
        });
        totals.overall_score += result.evaluation.overall_score || 0;
    });

    const averages = {};
    Object.keys(totals).forEach(key => {
        averages[key] = (totals[key] / validResults.length).toFixed(2);
    });

    return averages;
}

// Main execution with performance monitoring
async function main() {
    const startTime = Date.now();

    try {
        console.log('=== FAQ Relevance Evaluation Started ===\n');

        // Choose processing method
        const useAdvancedBatching = 'true';

        let results;
        if (useAdvancedBatching) {
            results = await processAllFAQsWithLimit();
        } else {
            results = await processAllFAQsConcurrent();
        }

        await saveResults(results);

        const endTime = Date.now();
        const totalTime = endTime - startTime;

        console.log('\n=== Processing Complete ===');
        console.log(`Total FAQs processed: ${results.length}`);
        console.log(`Successful evaluations: ${results.filter(r => r.success).length}`);
        console.log(`Failed evaluations: ${results.filter(r => !r.success).length}`);
        console.log(`Total processing time: ${(totalTime / 1000).toFixed(2)} seconds`);
        console.log(`Average time per FAQ: ${(totalTime / results.length).toFixed(2)}ms`);

    } catch (error) {
        console.error('Script execution failed:', error);
        process.exit(1);
    }
}

// Run the script
if (require.main === module) {
    main();
}

module.exports = {
    fetchFAQsFromMongo,
    evaluateRelevance,
    processAllFAQsConcurrent,
    processAllFAQsWithLimit,
    processSingleFAQ
};